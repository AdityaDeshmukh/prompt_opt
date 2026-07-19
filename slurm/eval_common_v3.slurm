#!/bin/bash
###############################################################################
# Confirmatory ranking eval: every v2 run that reached step >= 1200 is
# evaluated at EXACTLY step 1200 (fair matched-step comparison) on the test
# split, fp32 scorers, vLLM task-LM backend with a fixed seed.
# 4-way array; task i handles runs where (index % 4) == i.
# Usage: sbatch slurm/eval_common_v3.slurm
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_v3_evalc
#SBATCH --partition=secondary
#SBATCH --constraint="A100|H100|h100|A40|RTXA6000"
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

STEP=1200
i=0
FAILED=0
for RUN_DIR in /u/ad11/prompt_opt/outputs/v2/*/; do
    RUN=$(basename ${RUN_DIR})
    CKPT=${RUN_DIR}ckpt/ckpt.step.${STEP}.pth
    [ -f "$CKPT" ] || continue
    idx=$((i % 4)); i=$((i + 1))
    [ "$idx" -ne "$SLURM_ARRAY_TASK_ID" ] && continue
    [ -f "${RUN_DIR}test/output.step.${STEP}.json" ] && \
        { echo "skip ${RUN} (already evaluated)"; continue; }
    ALGO=$(echo ${RUN} | sed -E 's/^v2(lora)?_//; s/_seed[0-9]+$//')
    EXTRA=""
    [[ ${RUN} == v2lora_* ]] && EXTRA="adaptor_type=hyperlora"
    echo "[$(date)] evaluating ${RUN} @ step ${STEP}"
    $PY run_eval.py \
            algo=${ALGO} ${EXTRA} \
            report_to_wandb=false \
            device=cuda:0 device_scorer=cuda:0 \
            task_lm_backend=vllm vllm_seed=0 \
            scorer_dtype=float32 \
            save_dir=${RUN_DIR} \
            checkpoint_path=${CKPT} \
        || { echo "[$(date)] EVAL FAILED: ${RUN}"; FAILED=$((FAILED+1)); }
done
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done (${FAILED} failures)"
[ "$FAILED" -gt 0 ] && exit 1
exit 0
