#!/bin/bash
###############################################################################
# 30-step timing benchmark of the refactored pipeline (no eval/save/wandb).
# Usage: sbatch slurm/benchmark.slurm
###############################################################################
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_bench
#SBATCH --partition=csl
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err

set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prompt_opt
cd /u/ad11/prompt_opt
# call the env python directly; `conda activate` does not switch
# the interpreter in a non-interactive SLURM shell here
PY=/u/ad11/miniconda3/envs/prompt_opt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MEM_DEBUG=1

nvidia-smi
for ALGO in rrebel_l1_std grpo; do
    echo "===== benchmarking ${ALGO} ====="
    /usr/bin/time -v $PY run_tst_multi_obj.py \
            algo=${ALGO} \
            random_seed=0 \
            device=cuda:0 device_scorer=cuda:0 \
            max_train_steps=30 \
            do_eval=false do_save=false report_to_wandb=false \
            print_every=10 \
            save_dir=/u/ad11/prompt_opt/outputs/bench_${ALGO} 2>&1 | tee /u/ad11/prompt_opt/logs/bench_${ALGO}.full | tail -40
done
