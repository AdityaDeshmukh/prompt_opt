#!/bin/bash
###############################################################################
# v4 frontier regeneration: 4 arms x seed 0 = 4 single-GPU runs, 12000 steps.
#
# WHY THIS EXISTS
# The v3 campaign completed (all 15 runs, 12000 steps) but /scratch was purged
# around 2026-09-03..05: every ckpt/ and eval/ under outputs/v3/ is now empty
# (0 *.pth, 0 outputs.step.*.json anywhere under /scratch/ad11). The scalar
# results survive on wandb, but score_trainer.evaluate() collapses the lambda
# sweep before logging (mean over lmbdas), so wandb has only lambda-AVERAGED
# content/style. The per-lambda arrays needed for the content-vs-sentiment
# tradeoff curve existed ONLY in the purged eval JSONs. This rerun regenerates
# them at one seed per arm.
#
# DELIBERATELY IDENTICAL to train_v3.slurm: algo args, eval_steps, max steps,
# env, partition strategy, group size, batch sizes. Do not "improve" anything
# here -- the point is that v4 seed0 be comparable to v3 seed0.
#
# DELIBERATELY DIFFERENT:
#   - run/wandb ids are v4_* and outputs go to outputs/v4/. Resuming the v3
#     wandb ids would append new history to the FINISHED v3 runs and destroy
#     the only surviving record of the completed campaign.
#   - archive_cycle(): after every cycle, eval JSONs (+ the newest checkpoint)
#     are copied to /u/ad11/prompt_opt/results/v4/ in HOME. This is the fix for
#     the root cause of the v3 data loss -- scratch is purgeable, so nothing
#     that a paper depends on may live only there.
#
# Usage: sbatch slurm/train_v4.slurm             (all 4)
#        sbatch --array=2 slurm/train_v4.slurm   (one run)
#        touch /u/ad11/prompt_opt/STOP_V4        (stop all chains at cycle end)
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
# Right-sized 2026-09-18. `--ntasks-per-node=16` requested 16 TASKS (=16 CPUs)
# while the script runs a single python process, so 15 slots were pure waste --
# and on ccc0324, which has CPUTot=16, that request took the WHOLE node: it
# blocked the node's second GPU and could only start once all 16 CPUs were
# idle. It also pulled ~1 TB of memory, because with no --mem a full-CPU
# request gets the node's full memory under CR_CPU_MEMORY.
# Measured need: 0.4-0.6 cores average, MaxRSS 6.6 GB.
# A GPU cannot be requested without CPUs -- a Slurm allocation is CPUs + memory
# + GRES, and the process needs a core to run on -- so the fix is to ask for a
# few, not to ask for none.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=tst_v4
#SBATCH --partition=secondary
# Node eligibility has a HARD FLOOR and a SOFT CEILING.
#
# FLOOR (--exclude): every GPU with <40 GB VRAM. gpt2-xl under vLLM with
# max_gen_batch_size=400 does not fit in 24 GB, and V100/T4 lack native bf16
# (~12x slower -> a cycle never reaches the 150-step checkpoint and the chain
# dies silently). Adding `scavenger` exposes five QuadroRTX6000 nodes (24 GB)
# and two more V100 nodes that `secondary` alone never offered -- a probe job
# submitted 2026-09-18 landed on ccc0232 (Quadro RTX 6000, 24 GB) in 25 s, so
# these MUST be excluded or the chain will happily schedule onto a node it
# cannot run on.
#   V100          ccc0089 ccc0090 ccc0215
#   TeslaT4       ccc0286 ccc0287
#   A30 (24 GB)   ccc0478
#   QuadroRTX6000 ccc0232-ccc0236 (24 GB)
#SBATCH --exclude=ccc0089,ccc0090,ccc0215,ccc0286,ccc0287,ccc0478,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
#
# CEILING (--prefer): a SOFT constraint -- take a fast GPU if one is free, else
# take any eligible one rather than idle. Measured s/step over 27,750 steps:
#   H100 16.7 | L40S 27-30 | A100 29 | RTXA6000 44 | A40 88
# A hard --constraint=H100 was measured to still be queued after 4 min while an
# unconstrained scavenger job started in 25 s, so the ceiling must stay SOFT.
# L40S has no standalone feature (only AE7763_100g_1T_L40S), so it is reachable
# through the floor but not nameable here. Feature names are CASE-SENSITIVE and
# this cluster spells H100 both ways on different nodes -- both are required.
#SBATCH --prefer=H100|h100|h200|rtx6000b
# the EXIT trap owns chain continuation; requeue would create a second
# concurrent chain for the same run and corrupt its checkpoints.
#SBATCH --no-requeue
#SBATCH --array=0-3
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%A_%a.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%A_%a.err
#SBATCH --mail-user=ad11@illinois.edu
#SBATCH --mail-type=FAIL

cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0  # native torch sampler; flashinfer needs ninja+nvcc JIT
# HF cache lives on scratch (home quota is 100G). Set it EXPLICITLY rather than
# relying on the ~/.cache/huggingface symlink: the 2026-09 purge deleted the
# gpt2-xl weight blob while leaving its cache directory in place, so vLLM tried
# to re-download on a compute node and died with a confusing 404. Weights were
# restored 2026-09-09 (6.43 GB safetensors, snapshot 15ea56de...).
export HF_HOME=/scratch/ad11/hf_cache

MAX_STEPS=12000
MAX_NOPROGRESS_RETRIES=3
SEED=0
# Arms (trimmed from 5 on 2026-09-08 so the array would schedule; huber added
# back 2026-09-18 on request):
#   rrebel_l1_std  the best R-REBEL variant (tied with huber_std within noise)
#   grpo_ent       the DEFENSIBLE GRPO baseline. Plain `grpo` is dropped because
#                  the audit showed it never learns: arm mean 30.38 @500 ->
#                  30.19 @11500, KL pinned at the k3 ceiling, median 5 distinct
#                  tokens per cycle = one frozen 5-token prompt. See AUDIT.md.
#   grpo_baseref   NEW, and the experiment the audit says was missing. Plain
#                  GRPO but with grpo_ref_mode=base_lm, so the pinned KL anchor
#                  is the unadapted backbone LM instead of the initial policy.
#                  The v3 anchor was numerically UNIFORM (measured entropy
#                  10.824907 vs ln(50257)=10.824905), which makes beta*KL an
#                  entropy bonus rather than a trust region; base_lm measures
#                  7.672 nats, so the anchor finally exerts real force. This is
#                  the arm that answers "was your baseline actually GRPO?".
#   rrebel_huber_std  added 2026-09-18. In v3 it scored 39.25 against
#                  rrebel_l1_std's 39.47 -- a 0.22 gap, far inside the ~1-point
#                  noise floor, so the two were reported as TIED for best and
#                  neither can be called the winner. Needs no ALGO_ARGS: the
#                  dispatch already fixes d=huber with reward_std_scale=True,
#                  and huber_delta defaults to 1.0.
ARMS=(rrebel_l1_std grpo_ent grpo_baseref rrebel_huber_std)
T=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$T]}
ALGO=$ARM
ALGO_ARGS=""
if [ "$ARM" = "grpo_ent" ]; then
    ALGO=grpo
    # 0.04 = DeepSeekMath standard. grpo_ref_sync_steps=0 pins the anchor for
    # the whole run (TRL/verl/OpenRLHF behaviour); the v3 bug was re-deepcopying
    # it every update_steps=150, which zeroes the KL and its gradient at each
    # refresh so the anchor exerts no restoring force.
    ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.01"
elif [ "$ARM" = "grpo_baseref" ]; then
    ALGO=grpo
    ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.0 grpo_ref_mode=base_lm"
fi
RUN=v4_${ARM}_seed${SEED}
WANDB_NAME=${RUN}
# Walltime regime. V4_TIMEOUT must stay comfortably INSIDE the walltime so the
# EXIT trap gets to archive and resubmit before SLURM kills the allocation
# (4h job -> 13200s leaves 20 min; a 7-day job would need 597600s, leaving 2h).
# Longer single allocations exist (scavenger caps at 24h, csl at 7 days) but the
# measured wait rises steeply with the request: scavenger GPU jobs asking >12h
# waited a median 12.9h once one user's 2,529 one-minute jobs are excluded,
# versus 0.02h at <=1h. Long requests are hard to backfill, and 88% of ALL job
# starts on this cluster come from the backfill scheduler (sdiag: 2,943 of
# 3,336 starts in one 8h window). Short jobs that chain beat one long job.
TIMEOUT_S=${V4_TIMEOUT:-13200}
WALL=${V4_WALL:-04:00:00}
# CPUs per task. Measured 2026-09-18 over six cycles: TotalCPU 1:27-2:15 against
# 3:40 elapsed = 0.4-0.6 cores average, MaxRSS 6.6 GB. The old 16-CPU request
# was not merely over-provisioned: ccc0324 has CPUTot=16, so the job took that
# node ENTIRELY (blocking its second GPU and requiring a fully idle node), and
# on the 96-CPU/8-GPU H100 nodes 16 exceeds the 12-CPU-per-GPU fair share. That
# is a first-order cause of the 35-42% duty cycle. No effect on step time:
# generation is ~97% of a step and runs on the GPU.
NTASKS=${V4_NTASKS:-4}   # cpus-per-task
# PRESUBMIT=1 queues the NEXT cycle at the START of this one, gated on this job
# finishing (--dependency=afterany). NOTE what this does and does not buy:
# PriorityFlags is empty on this cluster (no ACCRUE_ALWAYS), so a dependency-held
# job is NOT eligible and therefore accrues NO age priority while it waits. The
# gain is only that the successor is already in the queue the instant the parent
# ends, instead of paying submit-and-be-noticed latency afterwards. Age is worth
# having: PriorityWeightAge=1000 over PriorityMaxAge=7d is the ONLY differentiator
# between our jobs, since FairShare/QOS/JobSize weights are all 0 and every
# partition has PriorityJobFactor=1 (flat 2000). That is also exactly why a
# pending job must be edited with `scontrol update`, NEVER cancelled and
# resubmitted -- cancelling throws away every point of accrued age.
# Measured duty cycle under the old end-of-cycle resubmit: 35-42%, i.e. ~60% of
# wall time was spent queued rather than training.
# When set, resubmit_decision must NOT also resubmit -- two successors, one of
# them undependent, would run concurrently on the same checkpoint dir.
PRESUBMIT=${V4_PRESUBMIT:-0}
# PARTITION CHOICE -- rewritten 2026-09-18 after measuring the whole campaign.
#
# Was: "secondary,csl". Now: "secondary,scavenger". Evidence:
#
# 1. csl is 2 nodes (ccc0435/0436) and BOTH are drain/drng as of 2026-09-18, so
#    it contributes zero capacity. It also produced 6 of our 6 preemptions --
#    csl is the only partition we used with AllowQos=ALL, and scav_high (prio
#    1000) preempts normal there. secondary's preemption rate is 0.34%.
# 2. Slurm REORDERS the partition list by PriorityTier (csl/scavenger are tier
#    3, secondary is tier 1), so "secondary,csl" is stored as "csl,secondary" --
#    verified on a fresh submit. With csl first, `sbatch --test-only` estimated
#    a start 17 DAYS later than the same job without csl, reproducibly (3/3).
# 3. scavenger is 63 nodes at tier 3 with a 24h cap, open to all accounts, and
#    it holds hardware secondary never offered us: H200 x8, RTX6000B x8 on four
#    nodes, plus more H100/L40S.
#
# Measured wait (per-user median, then median ACROSS users, so no single user's
# job flood can skew it; GPU jobs, 2026-09-11..18):
#     partition   <=1h wall   <=2h    =4h wall
#     scavenger      0.02h    0.11h     2.67h
#     secondary      1.25h    1.14h     2.27h
# and a real 10-min probe job started on scavenger in 25 SECONDS.
#
# Our own campaign for comparison: 410.7h queued vs 241.9h running = 37% duty
# cycle, and 139.8 of those 241.9 GPU-hours (58%) went to ccc0324, the SLOWEST
# eligible GPU, because "any GPU" resolves to the least-contended node and the
# least-contended node is the least desirable one. Hence --prefer above.
#
# NOT changed: the 4h walltime. The =4h bucket is the worst cell in the table
# (secondary 2.27h vs 0.44h at <=3h), so shortening cycles is the next big win
# -- but `save_steps=150` plus a hard `timeout` kill means work since the last
# checkpoint is lost, and per AUDIT.md the Adam state is NOT checkpointed, so
# every extra resubmit is another optimizer warm restart. Shortening the cycle
# changes training dynamics and is therefore a deliberate experiment, not a
# scheduling tweak. Fix optimizer-state checkpointing first, then shorten.
#
# PART is deliberately NOT carried in --export: sbatch --export takes a
# COMMA-separated list, so exporting V4_PART="secondary,scavenger" truncates it
# to "secondary" and silently costs the chain its scavenger eligibility.
PART=${V4_PART:-secondary,scavenger}
# Self-heal: a cycle launched by a PRE-2026-09-18 parent is submitted with that
# parent's in-memory PART ("secondary,csl"), which overrides the #SBATCH default
# above for that one cycle. Rewrite it here so the chain converges on its own
# rather than needing a human to notice. See the PARTITION CHOICE block for why
# csl is out: zero usable nodes and 6 of our 6 preemptions.
case ",$PART," in
    *,csl,*) echo "[$(date)] ${RUN}: inherited a csl partition list (${PART}) from a pre-2026-09-18 parent; resetting to secondary,scavenger"
             PART=secondary,scavenger ;;
esac

OUT_ROOT=/scratch/ad11/prompt_opt/outputs/v4
RUN_DIR=${OUT_ROOT}/${RUN}
CKPT_DIR=${RUN_DIR}/ckpt
EVAL_DIR=${RUN_DIR}/eval
# HOME archive: purge-proof copy of everything the paper depends on.
ARCHIVE_DIR=/u/ad11/prompt_opt/results/v4/${RUN}
STOP_FILE=/u/ad11/prompt_opt/STOP_V4
RETRY_FILE=${RUN_DIR}/.noprogress_count
mkdir -p "${RUN_DIR}" "${ARCHIVE_DIR}/eval"

if [ -f "$STOP_FILE" ]; then
    echo "[$(date)] ${RUN}: STOP_V4 present at start - exiting without running"
    exit 0
fi

latest_step() {  # highest checkpoint step, or 0
    local f; f=$(ls -1v ${CKPT_DIR}/ckpt.step.*.pth 2>/dev/null | tail -1)
    [ -z "$f" ] && { echo 0; return; }
    basename "$f" | sed -E 's/ckpt\.step\.([0-9]+)\.pth/\1/'
}

archive_cycle() {
    # Copy the eval JSONs into home. cp -u = only changed/new files.
    #
    # CHECKPOINTS ARE NOT COPIED HERE. ScoreTrainer._mirror_checkpoint does it
    # on the checkpoint-save event itself (ckpt_mirror_dir below), which is
    # strictly better: it fires every save_steps=150 rather than once per 4h
    # cycle, and it survives a SIGKILL/node-failure that would skip this trap.
    # Doing it in both places would just keep two ~750MB copies per run in a
    # 100G-quota home.
    local n=0
    if [ -d "$EVAL_DIR" ]; then
        cp -u ${EVAL_DIR}/outputs.step.*.json "${ARCHIVE_DIR}/eval/" 2>/dev/null
        n=$(ls -1 ${ARCHIVE_DIR}/eval/outputs.step.*.json 2>/dev/null | wc -l)
    fi
    # drop any checkpoint left by the pre-mirror version of this script
    ls -1v ${ARCHIVE_DIR}/ckpt.step.*.pth 2>/dev/null | xargs -r rm -f
    echo "[$(date)] ${RUN}: archived ${n} eval JSONs -> ${ARCHIVE_DIR}"
    echo "[$(date)] ${RUN}: mirrored checkpoint: $(cat ${ARCHIVE_DIR}/ckpt.latest.step 2>/dev/null || echo none)"
}

START_STEP=$(latest_step)

presubmit_next() {
    # Queue the successor NOW, dependent on this job ending for ANY reason
    # (completion, timeout, preemption, crash), so the chain is self-healing.
    # --kill-on-invalid-dep=yes cleans it up if this job is cancelled outright.
    local jid
    jid=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
            --kill-on-invalid-dep=yes \
            --partition="$PART" --time="$WALL" --cpus-per-task="$NTASKS" \
            --export=ALL,V4_WALL="$WALL",V4_TIMEOUT="$TIMEOUT_S",V4_NTASKS="$NTASKS",V4_PRESUBMIT=1 \
            --array=${T} /u/ad11/prompt_opt/slurm/train_v4.slurm 2>&1)
    if [ $? -eq 0 ]; then
        echo "[$(date)] ${RUN}: pre-submitted successor ${jid} (afterany:${SLURM_JOB_ID})"
    else
        echo "[$(date)] ${RUN}: PRE-SUBMIT FAILED (${jid}); falling back to end-of-cycle resubmit"
        PRESUBMIT=0
    fi
}

# Guards that used to live only at the END of a cycle. With pre-submit the
# successor is already queued before the outcome is known, so it has to be able
# to stand itself down.
if [ "$START_STEP" -ge "$MAX_STEPS" ]; then
    echo "[$(date)] ${RUN}: already at ${START_STEP} >= ${MAX_STEPS} - nothing to do"
    # only hand off to eval if the predecessor's trap did not (e.g. SIGKILL)
    if [ ! -f "${RUN_DIR}/test/output.step.${MAX_STEPS}.json" ]; then
        sbatch --array=${T} /u/ad11/prompt_opt/slurm/eval_v4.slurm \
            && echo "[$(date)] ${RUN}: submitted final eval (predecessor had not)"
    fi
    exit 0
fi
_retries=$(cat "$RETRY_FILE" 2>/dev/null || echo 0)
if [ "$_retries" -gt "$MAX_NOPROGRESS_RETRIES" ]; then
    echo "[$(date)] ${RUN}: no-progress budget already exhausted (${_retries}) - standing down"
    exit 0
fi
[ "$PRESUBMIT" = "1" ] && presubmit_next

resubmit() {  # sbatch with time-bounded retries (rides out a slurmctld outage)
    local deadline=$(( $(date +%s) + 900 ))   # 15 min
    local wait=30 k=0
    while [ "$(date +%s)" -lt "$deadline" ]; do
        k=$((k + 1))
        sbatch --partition="$PART" --time="$WALL" --export=ALL,V4_WALL="$WALL",V4_TIMEOUT="$TIMEOUT_S" --array=${T} /u/ad11/prompt_opt/slurm/train_v4.slurm && return 0
        echo "[$(date)] ${RUN}: sbatch attempt ${k} failed, retrying in ${wait}s"
        sleep "$wait"
        wait=$(( wait * 2 )); [ "$wait" -gt 120 ] && wait=120
    done
    echo "[$(date)] ${RUN}: ALL sbatch attempts failed over 15min - chain ends here"
    date +"%Y-%m-%dT%H:%M:%S%z chain died: sbatch unreachable for 15min" \
        > "${RUN_DIR}/.chain_died"
    return 1
}

resubmit_decision() {
    local end_step retries
    end_step=$(latest_step)
    archive_cycle
    echo "[$(date)] ${RUN}: cycle ended at step ${end_step} (started ${START_STEP})"
    if [ -f "$STOP_FILE" ]; then
        echo "[$(date)] ${RUN}: STOP_V4 present - not resubmitting"; return
    fi
    if [ "$end_step" -ge "$MAX_STEPS" ]; then
        echo "[$(date)] ${RUN}: reached ${MAX_STEPS} - done"
        # hand off to the final fixed-seed per-lambda eval
        sbatch --array=${T} /u/ad11/prompt_opt/slurm/eval_v4.slurm \
            && echo "[$(date)] ${RUN}: submitted final eval" \
            || echo "[$(date)] ${RUN}: final eval sbatch FAILED - run eval_v4.slurm by hand"
        return
    fi
    if [ "$end_step" -gt "$START_STEP" ]; then
        rm -f "$RETRY_FILE"
        echo "[$(date)] ${RUN}: progress ${START_STEP}->${end_step}"
        [ "$PRESUBMIT" = "1" ] && { echo "[$(date)] ${RUN}: successor already queued"; return; }
        resubmit
        return
    fi
    retries=$(cat "$RETRY_FILE" 2>/dev/null || echo 0)
    retries=$((retries + 1))
    if [ "$retries" -le "$MAX_NOPROGRESS_RETRIES" ]; then
        echo "$retries" > "$RETRY_FILE"
        echo "[$(date)] ${RUN}: NO progress (retry ${retries}/${MAX_NOPROGRESS_RETRIES})"
        [ "$PRESUBMIT" = "1" ] && { echo "[$(date)] ${RUN}: successor already queued"; return; }
        resubmit
    else
        echo "[$(date)] ${RUN}: no progress after ${MAX_NOPROGRESS_RETRIES} retries - GIVING UP (investigate!)"
    fi
}
trap resubmit_decision EXIT

CKPT_ARG=""
[ "$START_STEP" -gt 0 ] && CKPT_ARG="checkpoint_path=${CKPT_DIR}/ckpt.step.${START_STEP}.pth"
echo "[$(date)] ${RUN}: arm=${ARM} algo=${ALGO} args='${ALGO_ARGS}' part=${PART} wall=${WALL} timeout=${TIMEOUT_S}s resuming from step ${START_STEP} (target ${MAX_STEPS})"

timeout $TIMEOUT_S $PY run_tst_multi_obj.py \
        algo=${ALGO} \
        random_seed=${SEED} \
        device=cuda:0 device_scorer=cuda:0 \
        task_lm_backend=vllm \
        eval_steps=500 \
        run_name=${WANDB_NAME} \
        save_dir=${RUN_DIR} \
        ckpt_mirror_dir=${ARCHIVE_DIR} \
        ${ALGO_ARGS} \
        ${CKPT_ARG}
rc=$?
# 124 = `timeout` ended a still-training cycle: expected, not a failure.
[ "$rc" -eq 124 ] && exit 0
exit $rc
