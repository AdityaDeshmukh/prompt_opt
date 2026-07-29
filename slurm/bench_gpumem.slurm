#!/bin/bash
# Does giving vLLM more of the GPU speed up the step? Runs ~10 real training
# steps at several vllm_gpu_memory_utilization values on one big (>=80GB) card
# and prints per-step wall time. If step time is flat, we're compute-bound and
# 0.35 is already fine; if it drops, bump the config (card-aware).
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|h100"
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=tst_v3_gmem
#SBATCH --partition=secondary
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err

cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for GMU in 0.35 0.55 0.75; do
  echo "===== vllm_gpu_memory_utilization=${GMU} ====="
  timeout 900 $PY run_tst_multi_obj.py \
        algo=rrebel_l1_ent random_seed=0 \
        device=cuda:0 device_scorer=cuda:0 \
        task_lm_backend=vllm vllm_gpu_memory_utilization=${GMU} \
        max_train_steps=10 do_eval=false do_save=false report_to_wandb=false \
        save_dir=/scratch/ad11/prompt_opt/outputs/bench_gmem_${GMU} \
    2>&1 | awk '/^Step:/{print strftime("[%H:%M:%S]"),$0; fflush()}'
done
echo "gpumem bench done"