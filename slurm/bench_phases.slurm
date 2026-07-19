#!/bin/bash
###############################################################################
# One-off per-phase timing harness (does NOT touch the live training code path;
# the python only imports the modules). Single short secondary job.
# Usage: sbatch slurm/bench_phases.slurm
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=tst_bench_phase
#SBATCH --partition=secondary
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate prompt_opt
cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

$PY /u/ad11/prompt_opt/slurm/bench_phases.py
