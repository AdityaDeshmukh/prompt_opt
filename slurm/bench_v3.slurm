#!/bin/bash
# 15-step timed v3 benchmark on a csl L40S (same card class as the v2
# measurements: 69.5s gen @mgbs128 / 30.4s gen @mgbs512 + ~9s rest).
# Timestamps every line so per-step wall time is recoverable.
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=tst_v3_bench
#SBATCH --partition=csl
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err

cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name --format=csv,noheader

$PY run_tst_multi_obj.py \
        algo=rrebel_l1_ent \
        random_seed=0 \
        device=cuda:0 device_scorer=cuda:0 \
        task_lm_backend=vllm \
        max_train_steps=15 \
        do_eval=false do_save=false report_to_wandb=false \
        print_every=100 \
        save_dir=/scratch/ad11/prompt_opt/outputs/bench_v3 \
  2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' | grep -E "^\[[0-9:]+\] (Step:|GPU|NVIDIA)"
echo "bench done"