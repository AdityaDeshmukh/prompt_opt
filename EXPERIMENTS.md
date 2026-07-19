# Experiment design: GRPO vs R-REBEL variants

## v3 campaign (2026-07-19, current)

The v2 campaign died on 2026-07-17: home directory hit its 103G hard quota,
every running job crashed (log files could not even be created), and the
self-resubmit guard — which required progress — correctly refused to
resubmit, silently ending all 25 chains at steps 150-3150.

v3 changes, in order of importance:

1. **vLLM task-LM backend** (`task_lm_backend=vllm`, env `prompt_opt_v3`:
   vllm 0.25.1 / torch 2.11 / transformers 5.14). Task-LM generation was
   measured at 97.6% of a training step; vLLM replaces the chunked
   `model.generate` loop with continuous batching + paged KV cache +
   copy-on-write prompt sharing for the 50 samples/prompt. The transformers
   path survives as `task_lm_backend=hf` (auto-fallback when vllm is not
   installed).
2. **Outputs on /scratch** (`/scratch/ad11/prompt_opt/outputs/`, home holds a
   symlink) — removes the quota failure mode entirely. NOTE: campus scratch
   is subject to purge of old files; archive final checkpoints + eval JSONs
   when the campaign ends.
3. **Crash-proof resubmit chains** (`slurm/train_v3.slurm`): the resubmit
   decision moved into a bash EXIT trap (fires on crash and scancel TERM,
   not just clean exit), with a 3-cycle no-progress retry budget and a
   `STOP_CAMPAIGN` kill-switch file.
4. **Simplified matrix**: GRPO + the top-3 R-REBEL variants from the v2
   ranking, MLP adaptor only, 3 seeds = 12 runs, 12k steps.

### v2 ranking (dev evals at matched steps; confirmatory test eval at step 1200)

Preliminary ranking from the training-time dev evals (matched steps,
constrained score): `rrebel_l1_ent` most robust (best at 2000+, 3 seeds),
`rrebel_huber_std` top at 500-1500 (thin coverage), `rrebel_l1_std` solid,
`rrebel_l2` consistently last among R-REBEL. GRPO on the MLP adaptor scores
~4x below every R-REBEL variant and degrades with training — a headline
result. Confirmatory eval: `slurm/eval_common_v3.slurm` (test split, fp32
scorers, all runs at exactly step 1200); ranking script:
`analysis/rank_variants.py`.

---

# v2 design notes (historical)

## Goal

Compare four RL algorithms for training the λ-conditioned hypernetwork on the
Yelp negative→positive prompt-optimization benchmark (see `draft.pdf`):
maximize sentiment (style) score subject to a content-preservation constraint,
with the operating point selected by λ (τ in the paper).

| algo key        | Loss                                                     | Provenance |
|-----------------|----------------------------------------------------------|------------|
| `grpo`          | GRPO: group-standardized advantage, clipped surrogate    | corrected `grpo_loss` (ε-guarded std, β from config) |
| `rrebel_l2`     | R-REBEL, squared d (= REBEL with all pairs)              | cleaned `drgo_loss` |
| `rrebel_l1_std` | R-REBEL, ℓ1 d + per-group std reward scaling             | the paper's best config (std was previously commented out in `l1`) |
| `rrebel_l1_ent` | R-REBEL, ℓ1 + std + entropy bonus (`ent_coef=0.01`)      | rl_env `RREBEL_IMPROVEMENTS.md` finding #1 (fixes ℓ1 "dead groups") |
| `rrebel_huber_std` | R-REBEL, Huber d + per-group std (`huber_delta=1.0`)  | rl_env study: robust middle ground (Acrobot 4/5 vs ℓ1 2/5); added 2026-07-13 |

All four now share **identical** reward scaling (`score_scale * score_scaler_fnc(λ)`),
ε-guarded std denominators, one rollout per optimizer step (the old
retry-until-loss<1000 loop is gone), the same sampling distribution
(`explore=false` by default for every algo), and the same 150-step lagged
reference (GRPO with `grpo_beta=0` skips the reference pass entirely — it is
KL-free by construction, which is disclosed as part of the comparison).

Fairness fixes applied relative to the old runs (full audit in the workflow
report): retry-loop gradient accumulation, 0.1-scale inconsistency between
variants, missing std ε (NaN hazard), std-scaling absent from `l1`,
explore-flag mismatch between GRPO and R-REBEL launches.

## Arms

- **A (primary)** — `slurm/train_matrix.slurm`: MLP HyperNet head on frozen
  distilgpt2 (same architecture as all previous runs), task LM gpt2-xl.
  4 algos × 3 seeds = 12 single-GPU runs.
- **B (HyperLoRA)** — `slurm/train_hyperlora.slurm`: λ-conditioned LoRA
  (`adaptor_type=hyperlora`): a hypernetwork maps λ to per-sample rank-4
  A/B deltas on each block's `attn.c_attn`; logits come from the frozen
  lm_head. ~1.4M trainable params vs ~11M for the MLP head; policy equals
  the base LM exactly at init (B-heads zero-init). 4 algos × 2 seeds = 8 runs.
- **C (newer models, exploratory)**: the code is now backbone-generic
  (AutoModelForCausalLM, id-based decoding). Verified available and
  compatible with the pinned transformers 4.49: `Qwen/Qwen2.5-0.5B-Instruct`
  (policy), `Qwen/Qwen2.5-1.5B-Instruct` or `HuggingFaceTB/SmolLM2-1.7B-Instruct`
  (task LM); all ungated. Qwen3 needs transformers ≥4.51; Llama-3.2/gemma-2
  are gated (no HF token on this account). Changing models changes the
  benchmark — run as a separate arm after A/B, e.g.:
  `python run_tst_multi_obj.py algo=rrebel_l1_std adaptor_type=hyperlora \
   policy_lm=Qwen/Qwen2.5-0.5B task_lm=Qwen/Qwen2.5-1.5B-Instruct \
   pad_token='<|endoftext|>' ...`
  (set `HF_HOME=/scratch/ad11/hf` first — home quota is at 86G/100G).

## Shared protocol

batch 15 sources × `num_repeats=16` (group size G) per step, `num_samples=50`
task-LM samples per prompt, 12,000 steps (≈1 epoch of the 177k train split),
Adam lr 1e-4, grad-clip 10, ref refresh every 150 steps, λ ~ U[0,1] per source
per step. Constrained reward (unchanged):
`score = style if content ≥ 100λ else 0.01·(content − 100λ)`.

- During training: eval every 500 steps (10 dev sources × λ ∈ {0,…,0.9}).
- Final: `slurm/eval_matrix.slurm` evaluates the last checkpoint of every run
  on the 500-source test split with the λ sweep → per-λ (content, style)
  points → tradeoff curves via `plot2.py`.
- Report per algo: mean tradeoff curve over seeds, monotonicity of
  λ → (content, style) (the controllability claim), and hypervolume/AUC of
  the curve as a scalar summary.

## Performance changes (old ≈153 s/step → target ≤25 s/step)

1. Task-LM generation batched across prompts (`sample_generate_grouped`):
   rows sorted by per-source `max_new_tokens`, chunks of
   `max_gen_batch_size=400` rows, left padding, one `model.generate` per
   chunk (was: 240 sequential pipeline calls/step).
2. gpt2-xl in bf16 (`task_lm_dtype`), scorers in fp16 (`scorer_dtype`).
3. BERTScore + style classifier called once per step over all 12,000 outputs
   (`style_batch_size=256`), not 240 times over 50.
4. Policy rollout is id-based with an explicit attention mask (fixes the
   pad-attended KV-cache pollution and the token→string→retokenize desync),
   sampling vectorized over the batch.
5. Single GPU per run (`device=device_scorer=cuda:0`, ~28 GB peak in bf16),
   so the 8 L40S of one csl node run 8 experiments concurrently.
6. GRPO skips the reference teacher-forcing pass (β=0 ⇒ unused).
7. Per-step printing gated (`print_every=50`) — the old 240 lines/step
   produced 300+ MB logs per run.

Budget check: at ≤25 s/step, 12k steps ≤ 3.5 days — inside the 7-day csl
MaxTime with margin (`--time=4-00:00:00`), auto-resume from the latest
checkpoint is built into the array scripts. Run `slurm/benchmark.slurm`
first to measure the real step time (30 steps, ~15 min).

### 2026-07-15 profiling + generation batch-size fix

Per-phase timing of a full R-REBEL step (harness `slurm/bench_gen.py` /
`bench_phases.py`) showed task-LM generation is **97.6%** of a step; content
scoring 1.4%, style 0.9%, rollout+ref+backward <0.4%. So generation is the
*only* phase worth optimizing (BERTScore-dedup, rollout, and backward tuning
were measured and dropped as immaterial).

Two fixes applied to all four training scripts:
1. **`max_gen_batch_size` 128 → 400.** The array scripts had overridden it to
   128 for OOM-safety on small secondary cards; with `num_samples=50` that is
   only 2 prompts/`generate` call → 120 tiny calls/step. On an L40S the sweep
   is 128:69.5s → 256:34.9s → 512:30.4s (knee). 400 (8 rows/chunk, 30 calls)
   captures ~2.2× on generation ≈ ~2× on the whole step, and matches the
   full-pipeline mem-proven value (~28 GB peak). Comparability-safe: only the
   RNG/batching of the 50-sample reward estimate changes, not the model,
   sampling dist, sample count, reward, or any hyperparameter; applied to all
   25 runs equally so the cross-algo comparison is unaffected.
2. **Fast-card selection.** bf16 is emulated (≈12× slower) on the V100
   (Volta); a V100-bound run does ~34 steps/4 h < the 150-step checkpoint
   interval, so it never checkpoints, the self-resubmit guard sees "no
   progress", and the run dies silently. Secondary-only scripts now
   `--constraint="A100|H100|h100|A40|RTXA6000"` (all ≥40 GB, native bf16;
   also drops the 24 GB A30, too small for mgbs=400). The multi-partition
   `train_matrix_cslok` (secondary,csl) can't whitelist features valid in both
   partitions, so it uses `--exclude=ccc0215,ccc0286,ccc0287,ccc0478` (V100,
   2×T4, A30) instead. Rolls out per-run on the next self-resubmit.

Remaining levers not taken (future / eval / newer-model arm): prefill-shared
decode (~1.4× more, prototype in `bench_gen.py` but unverified for
correctness) and vLLM for the frozen task LM (potential 5–15× on generation,
heavy dependency). bf16 vs fp16 is a wash on modern cards (0.98×).

## Comparability with pre-refactor runs

New runs are **not bit-comparable** to the old curves, by design:
- the task LM samples in bf16 and training-time scorers run fp16 (both
  config-revertible: `task_lm_dtype`/`scorer_dtype: float32`); final
  reported curves from `slurm/eval_matrix.slurm` use fp32 scorers;
- the policy rollout fixes two real bugs in the old padded decoding
  (right-pad tokens attended in the KV cache, token→string→retokenize
  drift), so the policy's state distribution shifts slightly;
- defaults changed to match what the old SLURM scripts passed explicitly:
  `num_repeats` 15→16, `num_samples` 20→50, `style_batch_size` 32→256,
  `device_scorer` cuda:1→cuda:0.
To compare against an old checkpoint, re-evaluate it through
`slurm/eval_matrix.slurm` rather than reusing its old logged numbers.
`algo=grpo` now runs the corrected loss (gradient verified identical to the
legacy one up to the std-ε guard); `algo=grpo_legacy` keeps the old code.
Old root-level SLURM scripts moved to `slurm/legacy/`.

## Housekeeping

- Old SLURM logs (7.6 GB of `*.output`/`*.error`) deleted 2026-07-12.
- `outputs/` still holds ~37 GB of every-150-step checkpoints from old runs;
  prune to final checkpoints when the old runs are no longer needed.
- Old flat losses (`drgo*`, `l1`, `plgo*`, `kl`) remain in
  `losses/loss_functions.py` under legacy names (`grpo` → `grpo_legacy`);
  old checkpoints remain loadable (state-dict layout unchanged).
