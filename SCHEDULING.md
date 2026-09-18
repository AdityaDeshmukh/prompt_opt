# Efficient SLURM submission on the Illinois Campus Cluster

Written 2026-09-18 after auditing the whole v4 campaign. Every number here was
measured on this cluster, on this date; commands to re-derive each one are given
so they can be re-checked when the cluster changes.

## What the campaign actually cost

```
TOTAL QUEUE WAIT : 410.7 h   (n=86 jobs)
TOTAL RUN TIME   : 241.9 h
DUTY CYCLE       :  37.1 %
```

We spent **17 days of wall-clock queueing to buy 10 days of compute**. Worse,
the compute we bought was the slowest available:

```
NODE      GPU             cycles  steps   GPU-h  sec/step
ccc0324   RTXA6000            47  11400   139.5      44.1   <- 58% of all our GPU-hours
ccc0423   H100                 7   5400    25.7      17.1
ccc0435   L40S                 7   2550    21.5      30.3
ccc0436   L40S                 5   2850    21.2      26.8
ccc0424   H100                 6   2250    11.0      17.7
ccc0457   H100                 3   2250    11.0      17.6
ccc0351   A100                 2    900     7.3      29.4
ccc0333   A40                  1    150     3.7      88.0
TOTAL: 27,750 steps in 241.0 GPU-h => 31.3 s/step average
```

Reproduce: `sacct -u $USER -S <date> -X -P -o JobID,Submit,Start,End,NodeList,State`
cross-referenced with `logs/tst_v4_*.out` ("cycle ended at step N (started M)").

## The five root causes

### 1. "Any GPU" is a bias toward the SLOWEST GPU

This is the big one and it is counter-intuitive. `--gres=gpu:1` gets you the
*least contended* node, and on a heterogeneous cluster the least contended node
is the least desirable one. 58% of our GPU-hours went to a 44 s/step RTXA6000
while 17 s/step H100s sat in the same partition.

`sbatch --test-only` with a bare `--gres=gpu:1` resolves to **ccc0287, a Tesla
T4**. Add `--prefer=H100|h100|h200|rtx6000b` and the same request resolves to
**ccc0424, an H100, at the same estimated start time**.

Cluster-wide, asking for H100 specifically costs almost nothing in queue time:

```
GPU TYPE REQUESTED    n     median wait
(any)              2590          2.28 h
h100               1213          2.56 h     <- +0.3 h for 2.6x the throughput
```

**Use `--prefer` (soft), not `--constraint` (hard).** A hard `--constraint=H100`
probe was still queued after 4 minutes while an unconstrained scavenger probe
started in 25 seconds. `--prefer` takes a fast GPU when one is free and an
acceptable one otherwise. It IS validated -- `--prefer=NOSUCHFEATURE` is
rejected -- so a typo fails loudly rather than being silently ignored.

**Feature names are CASE-SENSITIVE and this cluster is inconsistent**:
`--gres=gpu:H100:1` works, `--gres=gpu:h100:1` fails. Nodes ccc0419-0424/0439
carry feature `H100`; ccc0453/0456/0457 carry `h100`. You need both spellings.
`L40S` is not a standalone feature at all (only `AE7763_100g_1T_L40S`).

### 2. Requesting the partition's maximum walltime is the worst choice

**88% of all job starts on this cluster come from the backfill scheduler**
(`sdiag`: 2,943 backfilled of 3,336 started in one 8h window; the main scheduler
hits `default_queue_depth` in 87% of its cycles). Backfill can only start your
job if it fits in a hole before a reservation -- so the longer you ask for, the
fewer holes fit you.

Per-user median wait, then median ACROSS users (so no single user's job flood
can skew it), GPU jobs, 2026-09-11..18:

```
wall      scavenger   secondary
<=1h          0.02 h     1.25 h
<=2h          0.11 h     1.14 h
<=3h             --      0.44 h
=4h (cap)     2.67 h     2.27 h   <- what we were requesting
```

Always compute this median-of-user-medians way. The naive pooled median said
scavenger 24h jobs wait 0.01h; that was one user's 2,529 jobs whose *median
runtime was 1.5 minutes*. Excluding them the real figure is **12.9 h**.

### 3. csl was a trap

- 2 nodes total (ccc0435/0436), **both drain/drng** as of 2026-09-18 = zero capacity.
- **6 of our 6 preemptions came from csl.** It is the only partition we used with
  `AllowQos=ALL`, and `scav_high` (priority 1000) preempts `normal` there.
  Preemption rates: csl 6.11%, scavenger 5.52%, secondary 0.34%.
- Slurm **reorders the partition list by PriorityTier** (csl and scavenger are
  tier 3, secondary is tier 1), so `--partition=secondary,csl` is stored as
  `csl,secondary` -- verified on a fresh submit. With csl first, `--test-only`
  estimated a start **17 days later** than the identical job without csl,
  reproducibly (3/3 runs).

Caveat, stated honestly: that 17-day figure is the estimator's worst case, not
a prediction. All 56 jobs we actually submitted as `secondary,csl` did start,
mean wait 5.73 h. csl is still not worth it.

### 4. `scavenger` existed the whole time and we never used it

63 nodes, PriorityTier 3, `AllowAccounts=ALL`, 24h cap, and hardware secondary
never offered us: H200 x8, RTX6000B x8 on four nodes, more H100 and L40S. A real
10-minute probe job **started in 25 seconds**.

The catch: it also exposes five QuadroRTX6000 nodes (24 GB) and two extra V100
nodes. Our probe landed on ccc0232, a 24 GB Quadro, which cannot hold gpt2-xl at
`max_gen_batch_size=400`. **Widening the partition list REQUIRES widening
`--exclude` in the same commit**, or the chain will cheerfully schedule onto a
node it cannot run on.

### 5. Oversized CPU/memory (fixed 2026-09-18)

`--ntasks-per-node=16` requested 16 CPUs for a single python process.

```
CPU Efficiency:    2.55% of 2-10:42:56 core-walltime
Memory Efficiency: 0.64% of 980.41 GB
```

ccc0324 has `CPUTot=16`, so the job took that node **entirely**, blocking its
second GPU and requiring a fully idle node. With no `--mem` and
`DefMemPerNode=UNLIMITED` under `CR_CPU_MEMORY`, it also held ~1 TB of RAM for a
6.4 GB process. Now `--ntasks=1 --cpus-per-task=4 --mem=48G`.

You cannot request a GPU without CPUs: an allocation is CPUs + memory + GRES and
the process needs a core. Ask for a few, not none.

Honest caveat: controlled for walltime, the cluster-wide wait-vs-CPU signal is
**not** monotone (5-8 CPUs had the lowest median wait), so this was a
node-blocking and citizenship problem more than a queue-time problem. It is not
the main reason we were slow. Causes 1 and 2 are.

Note that CPU starvation is real though: right now ccc0232/0233/0234/0236 have
**32 idle GPUs between them and 0 free CPUs** -- unreachable by anyone.

## Priority: age is the only thing you control

```
PriorityType            = priority/multifactor
PriorityWeightAge       = 1000      PriorityMaxAge = 7-00:00:00
PriorityWeightPartition = 2000      (flat: every partition has PriorityJobFactor=1)
PriorityWeightFairShare = 0
PriorityWeightQOS       = 0
PriorityWeightJobSize   = 0
PriorityFlags           =           (no ACCRUE_ALWAYS)
```

Everything except age is either zero or constant. So:

- **Never `scancel` + resubmit a pending job to change it. Use `scontrol update`.**
  Cancelling discards every point of accrued age. Editing keeps it:
  `scontrol update JobId=N Partition=... ExcNodeList=... NumCPUs=4 MinMemoryNode=49152`
- A `--dependency`-held job is **not eligible**, so with no `ACCRUE_ALWAYS` it
  accrues **no** age while held. Pre-submitting still helps -- the successor is
  queued the instant the parent ends -- but not because of priority.
- `scav_high` (priority 1000, preempts `normal`) is on our `csl` association but
  `scavenger` has `AllowQos=normal`, so it is usable only in the `csl` partition,
  whose two nodes are drained. Currently worthless; re-check if csl returns.

## The configuration this produced

```
#SBATCH --partition=secondary,scavenger
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --exclude=ccc0089,ccc0090,ccc0215,ccc0286,ccc0287,ccc0478,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
#SBATCH --prefer=H100|h100|h200|rtx6000b
```

Hard floor excludes every GPU under 40 GB; soft ceiling steers to fast ones
without ever idling for them. Before: `--test-only` resolved to ccc0287 (Tesla
T4). After: ccc0424 (H100), all four arms.

## What was deliberately NOT changed, and why

The 4h walltime. Section 2 says the 4h cap is the worst cell in the table, and
1-2h cycles on scavenger would be the single biggest remaining win
(~0.02-0.11 h median wait). But:

- `save_steps=150` plus a hard `timeout` kill means everything since the last
  checkpoint is lost when the cycle ends.
- Per `AUDIT.md`, **Adam state is never checkpointed** (`score_trainer.py` saves
  only `model_state_dict`), so every resubmit is an optimizer warm restart.
  Going from 4h to 1h cycles would roughly quadruple the number of warm restarts
  for the remainder of the run -- and only for the remainder, which is worse,
  because it would apply unevenly across arms that are at different steps.

So shortening the cycle is a change to **training dynamics**, not a scheduling
tweak, and it must not be slipped in mid-campaign. The correct order is:
checkpoint optimizer state first, then shorten cycles, then the scavenger
short-walltime cell becomes free money. Estimated at H100 speed and measured
waits: ~65 steps/h today vs ~200 steps/h at 1-2h cycles on scavenger.

## Checklist for the next campaign

1. `sinfo -o "%P %l %D %G"` -- find every partition you can use, and its cap.
   Check `AllowAccounts`/`AllowQos` with `scontrol show partition`; do not assume
   the partition you were told about is the only one.
2. Measure wait by (partition x walltime) as median-of-user-medians before
   committing. Check for one user flooding the bucket.
3. Benchmark s/step per GPU model. Set a hard `--exclude` floor on what cannot
   run, and a soft `--prefer` ceiling on what runs fast.
4. Request the walltime you need, never the partition cap.
5. Right-size CPUs/memory from `seff`, and check `CPUTot` on your target nodes --
   a request equal to `CPUTot` takes the whole node.
6. Verify with `sbatch --test-only` which NODE you resolve to, not just when.
7. Edit pending jobs with `scontrol update`; never cancel and resubmit.
8. Checkpoint optimizer state, so cycle length is a free scheduling variable.
