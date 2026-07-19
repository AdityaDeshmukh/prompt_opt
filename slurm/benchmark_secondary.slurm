#!/bin/bash
###############################################################################
# 30-step timing benchmark on the secondary partition (heterogeneous GPUs).
# Requests a generic GPU; max_gen_batch_size trimmed to 200 so it fits the
# smaller cards there (V100-16GB / A30-24GB) instead of OOMing.
# Usage: sbatch slurm/benchmark_secondary.slurm
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_bench_sec
#SBATCH --partition=secondary
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
            max_gen_batch_size=200 \
            max_train_steps=30 \
            do_eval=false do_save=false report_to_wandb=false \
            print_every=10 \
            save_dir=/u/ad11/prompt_opt/outputs/bench_sec_${ALGO} 2>&1 | tee /u/ad11/prompt_opt/logs/bench_sec_${ALGO}.full | tail -40
done
