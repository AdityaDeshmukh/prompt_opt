#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=tst_v3_smoke
#SBATCH --partition=secondary,csl
#SBATCH --exclude=ccc0215,ccc0286,ccc0287,ccc0478
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err

cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0  # native torch sampler; flashinfer needs ninja+nvcc JIT
nvidia-smi | head -12
$PY /u/ad11/prompt_opt/slurm/smoke_v3.py
