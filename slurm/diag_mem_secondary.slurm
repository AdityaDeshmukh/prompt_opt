#!/bin/bash
###############################################################################
# Memory diagnostic on one H100 (80GB, won't OOM) to measure the true single-
# GPU peak of the refactored pipeline; compare the reported peak against 48GB
# to decide single- vs dual-GPU on the L40S matrix.
# Usage: sbatch slurm/diag_mem_secondary.slurm
###############################################################################
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_diag_h100
#SBATCH --partition=secondary
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

run() {  # name  gen_batch
    echo "===================== CONFIG $1 (single-GPU, gen_batch=$2) ====================="
    $PY run_tst_multi_obj.py \
        algo=rrebel_l1_std random_seed=0 \
        device=cuda:0 device_scorer=cuda:0 \
        max_gen_batch_size=$2 \
        max_train_steps=4 \
        do_eval=false do_save=false report_to_wandb=false print_every=100 \
        save_dir=/u/ad11/prompt_opt/outputs/diagh_$1 2>&1 \
      | grep -E "\[MEM\]|Step:|OutOfMemory|out of memory|Traceback|RuntimeError" \
      | tee /u/ad11/prompt_opt/logs/diagh_$1.mem
    echo "(exit ${PIPESTATUS[0]})"
}

run single_b400 400
run single_b100 100
run single_b50  50
