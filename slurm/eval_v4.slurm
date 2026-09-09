#!/bin/bash
###############################################################################
# v4 final per-lambda eval: the figure-producing run.
#
# Evaluates each v4 arm's step-12000 checkpoint on the TEST split with fp32
# scorers and a FIXED vLLM seed, writing the per-lambda arrays
# (lmbdas / mean_scores / mean_contents / mean_styles) that
# analysis/plot_tradeoff.py turns into the content-vs-sentiment tradeoff curve.
#
# vllm_seed=0 (vs the campaign default of null) is deliberate and matters: with
# null, task-LM sampling is nondeterministic and a repeated eval moved a score
# by ~1.1 points, which is the same order as the gaps between the top R-REBEL
# variants. Training stays untouched; only this confirmatory eval is pinned.
#
# Auto-submitted per-arm by train_v4.slurm's EXIT trap when a run reaches
# 12000. Also runnable by hand for whatever has finished:
#   sbatch slurm/eval_v4.slurm            (all 3 arms; skips unfinished)
#   sbatch --array=2 slurm/eval_v4.slurm  (one arm)
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_v4_eval
#SBATCH --partition=secondary,csl
#SBATCH --exclude=ccc0215,ccc0286,ccc0287,ccc0478
#SBATCH --no-requeue
#SBATCH --array=0-2
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
# HF cache lives on scratch (home quota is 100G). Set it EXPLICITLY rather than
# relying on the ~/.cache/huggingface symlink: the 2026-09 purge deleted the
# gpt2-xl weight blob while leaving its cache directory in place, so vLLM tried
# to re-download on a compute node and died with a confusing 404. Weights were
# restored 2026-09-09 (6.43 GB safetensors, snapshot 15ea56de...).
export HF_HOME=/scratch/ad11/hf_cache

STEP=12000
SEED=0
ARMS=(rrebel_l1_std grpo_ent grpo_baseref)
T=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$T]}
ALGO=$ARM
ALGO_ARGS=""
# The loss/flags must match training, because ScoreLossModule is constructed
# from them even for a pure eval (a mismatched algo can change what gets built
# and, for the grpo arms, whether a ref pass is set up at all).
if [ "$ARM" = "grpo_ent" ]; then
    ALGO=grpo
    ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.01"
elif [ "$ARM" = "grpo_baseref" ]; then
    ALGO=grpo
    ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.0 grpo_ref_mode=base_lm"
fi
RUN=v4_${ARM}_seed${SEED}
RUN_DIR=/scratch/ad11/prompt_opt/outputs/v4/${RUN}
CKPT=${RUN_DIR}/ckpt/ckpt.step.${STEP}.pth
ARCHIVE_DIR=/u/ad11/prompt_opt/results/v4/${RUN}
mkdir -p "${ARCHIVE_DIR}/test"

if [ ! -f "$CKPT" ]; then
    # fall back to the newest checkpoint so a partially-trained arm still
    # yields a curve; the step is recorded in the output filename either way.
    CKPT=$(ls -1v ${RUN_DIR}/ckpt/ckpt.step.*.pth 2>/dev/null | tail -1)
    [ -z "$CKPT" ] && { echo "[$(date)] ${RUN}: no checkpoint at all - nothing to eval"; exit 0; }
    echo "[$(date)] ${RUN}: step ${STEP} ckpt absent, falling back to $(basename "$CKPT")"
fi

echo "[$(date)] ${RUN}: eval ${CKPT} (arm=${ARM} algo=${ALGO} args='${ALGO_ARGS}')"
$PY run_eval.py \
        algo=${ALGO} ${ALGO_ARGS} \
        random_seed=${SEED} \
        report_to_wandb=false \
        device=cuda:0 device_scorer=cuda:0 \
        task_lm_backend=vllm vllm_seed=0 \
        scorer_dtype=float32 \
        save_dir=${RUN_DIR} \
        checkpoint_path=${CKPT}
rc=$?

# Archive to HOME immediately -- this JSON is the tradeoff curve, and losing
# the v3 equivalents to a scratch purge is why v4 exists at all.
cp -u ${RUN_DIR}/test/output*.json "${ARCHIVE_DIR}/test/" 2>/dev/null \
    && echo "[$(date)] ${RUN}: archived test eval -> ${ARCHIVE_DIR}/test/"
ls -la "${ARCHIVE_DIR}/test/" 2>/dev/null

exit $rc
