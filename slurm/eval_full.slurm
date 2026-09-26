#!/bin/bash
###############################################################################
# v4 FULL test-split evaluation (2026-09-26).
#
# One array task = one (arm, checkpoint step, task-LM seed) line of
# slurm/eval_full_jobs.txt. Differences from eval_v4.slurm, and why:
#   * 20-point lambda grid (0.00..0.95) instead of 10 -> a smoother frontier.
#     Training samples lambda ~ U[0,1), so every grid point is in-distribution.
#   * 5 task-LM seeds at step 12000 -> Monte-Carlo error bars. The prompt is
#     greedy (deterministic given x, lambda); only the task LM's 50 samples per
#     prompt vary with the seed.
#   * per-(sentence, lambda) rows in the JSON (score / content / style /
#     feasible fraction) -> bootstrap CIs over the 500 test sentences and
#     PAIRED comparisons between arms (same sentences, same order).
#   * steps 3000/6000/9000 at seed 0 -> matched test-split learning curves.
# Writes to test_full/ so the eval_v4.slurm outputs in test/ are untouched.
#
#   sbatch --array=0-31 slurm/eval_full.slurm
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=tst_v4_full
#SBATCH --partition=secondary,scavenger
# floor/ceiling rationale: see train_v4.slurm and SCHEDULING.md
#SBATCH --exclude=ccc0089,ccc0090,ccc0215,ccc0286,ccc0287,ccc0478,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
#SBATCH --prefer=H100|h100|h200|rtx6000b
# a preempted eval is simply re-run from scratch: it has no state to corrupt
#SBATCH --requeue
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%A_%a.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%A_%a.err
#SBATCH --mail-user=ad11@illinois.edu
#SBATCH --mail-type=FAIL

cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME=/scratch/ad11/hf_cache

JOBS=${EVAL_FULL_JOBS:-/u/ad11/prompt_opt/slurm/eval_full_jobs.txt}
read -r ARM STEP SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS")
[ -z "$ARM" ] && { echo "no job line for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }
GRID=${EVAL_FULL_GRID:-"[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]"}
EXTRA=${EVAL_FULL_EXTRA:-}     # e.g. "max_size=50" for a smoke test
SUBDIR=${EVAL_FULL_SUBDIR:-test_full}

ALGO=$ARM; ALGO_ARGS=""
# must match training: ScoreLossModule is built from these even for eval
if [ "$ARM" = "grpo_ent" ]; then
    ALGO=grpo; ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.01"
elif [ "$ARM" = "grpo_baseref" ]; then
    ALGO=grpo; ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.0 grpo_ref_mode=base_lm"
fi
RUN=v4_${ARM}_seed0
RUN_DIR=/scratch/ad11/prompt_opt/outputs/v4/${RUN}
CKPT=${RUN_DIR}/ckpt/ckpt.step.${STEP}.pth
ARCHIVE_DIR=/u/ad11/prompt_opt/results/v4/${RUN}/${SUBDIR}
OUT=${RUN_DIR}/${SUBDIR}/output.step.${STEP}.seed${SEED}.json
mkdir -p "$ARCHIVE_DIR"
[ -f "$CKPT" ] || { echo "[$(date)] missing $CKPT"; exit 1; }
if [ -f "${ARCHIVE_DIR}/$(basename "$OUT")" ] && [ -z "$EXTRA" ]; then
    echo "[$(date)] already done: ${ARCHIVE_DIR}/$(basename "$OUT")"; exit 0
fi

echo "[$(date)] ${RUN} step=${STEP} seed=${SEED} node=${SLURMD_NODENAME} part=${SLURM_JOB_PARTITION} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
$PY run_eval.py \
        algo=${ALGO} ${ALGO_ARGS} \
        random_seed=0 report_to_wandb=false \
        device=cuda:0 device_scorer=cuda:0 \
        task_lm_backend=vllm vllm_seed=${SEED} \
        scorer_dtype=float32 \
        "eval_lmbdas=${GRID}" eval_subdir=${SUBDIR} \
        save_dir=${RUN_DIR} checkpoint_path=${CKPT} ${EXTRA}
rc=$?
[ $rc -eq 0 ] && [ -f "$OUT" ] && cp -f "$OUT" "$ARCHIVE_DIR/" && echo "[$(date)] archived -> ${ARCHIVE_DIR}/$(basename "$OUT")"
exit $rc
