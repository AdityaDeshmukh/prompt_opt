#!/bin/bash
###############################################################################
# Arm B: HyperLoRA adaptor — same 4 algos x 2 seeds, 8 single-GPU runs.
# Usage: sbatch slurm/train_hyperlora.slurm
#
# 24h wall + guarded self-resubmission across the maintenance window (see
# train_matrix.slurm for the mechanism).
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_v2_lora
#SBATCH --partition=secondary
#SBATCH --constraint="A100|H100|h100|A40|RTXA6000"
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%A_%a.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%A_%a.err
#SBATCH --mail-user=ad11@illinois.edu
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate prompt_opt
cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

MAX_STEPS=12000
ALGOS=(grpo rrebel_l2 rrebel_l1_std rrebel_l1_ent)
ALGO=${ALGOS[$((SLURM_ARRAY_TASK_ID % 4))]}
SEED=$((SLURM_ARRAY_TASK_ID / 4))
RUN=v2lora_${ALGO}_seed${SEED}
CKPT_DIR=/u/ad11/prompt_opt/outputs/v2/${RUN}/ckpt

latest_step() {
    local f
    f=$(ls -1v ${CKPT_DIR}/ckpt.step.*.pth 2>/dev/null | tail -1)
    [ -z "$f" ] && { echo 0; return; }
    basename "$f" | sed -E 's/ckpt\.step\.([0-9]+)\.pth/\1/'
}

START_STEP=$(latest_step)
CKPT_ARG=""
[ "$START_STEP" -gt 0 ] && CKPT_ARG="checkpoint_path=${CKPT_DIR}/ckpt.step.${START_STEP}.pth"
echo "[$(date)] ${RUN}: resuming from step ${START_STEP} (target ${MAX_STEPS})"

timeout 13500 $PY run_tst_multi_obj.py \
        algo=${ALGO} \
        adaptor_type=hyperlora \
        random_seed=${SEED} \
        device=cuda:0 device_scorer=cuda:0 \
        eval_steps=500 \
        max_gen_batch_size=400 \
        run_name=${RUN} \
        save_dir=/u/ad11/prompt_opt/outputs/v2/${RUN} \
        ${CKPT_ARG}

END_STEP=$(latest_step)
echo "[$(date)] ${RUN}: reached step ${END_STEP}"
if [ "$END_STEP" -lt "$MAX_STEPS" ] && [ "$END_STEP" -gt "$START_STEP" ]; then
    echo "[$(date)] ${RUN}: progress made (${START_STEP}->${END_STEP}), resubmitting array task ${SLURM_ARRAY_TASK_ID}"
    sbatch --array=${SLURM_ARRAY_TASK_ID} /u/ad11/prompt_opt/slurm/train_hyperlora.slurm
else
    echo "[$(date)] ${RUN}: not resubmitting (done, or no progress this cycle)"
fi
