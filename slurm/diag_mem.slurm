#!/bin/bash
###############################################################################
# Memory + timing diagnostic. Requests 2 GPUs and runs the pipeline three
# ways with per-phase GPU memory reporting (MEM_DEBUG=1):
#   A) single-GPU, both models on cuda:0, gen batch 100
#   B) single-GPU, both on cuda:0, gen batch 50 (smaller coincident peak)
#   C) two-GPU, policy cuda:0 / scorer cuda:1, gen batch 400 (original layout)
# Tells us definitively whether single-GPU fits and, if not, confirms 2-GPU.
# Usage: sbatch slurm/diag_mem.slurm
###############################################################################
#SBATCH --gres=gpu:L40S:2
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_diag
#SBATCH --partition=csl
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err

set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prompt_opt
cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt/bin/python
export MEM_DEBUG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv

run() {  # name  device  device_scorer  gen_batch
    echo "===================== CONFIG $1 ====================="
    $PY run_tst_multi_obj.py \
        algo=rrebel_l1_std random_seed=0 \
        device=$2 device_scorer=$3 \
        max_gen_batch_size=$4 \
        max_train_steps=4 \
        do_eval=false do_save=false report_to_wandb=false print_every=100 \
        save_dir=/u/ad11/prompt_opt/outputs/diag_$1 2>&1 \
      | grep -E "\[MEM\]|Step:|OutOfMemory|out of memory|Traceback|RuntimeError" \
      | tee /u/ad11/prompt_opt/logs/diag_$1.mem
    echo "(exit ${PIPESTATUS[0]})"
}

run A_single_b100 cuda:0 cuda:0 100
run B_single_b50  cuda:0 cuda:0 50
run C_dual_b400   cuda:0 cuda:1 400
