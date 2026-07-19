#!/bin/bash
# One short csl L40S benchmark run (Ada, native bf16; representative of the csl
# campaign cards). ~15 min; takes csl to at most 2 briefly (cap is 4).
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=tst_bench_gen
#SBATCH --partition=csl
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate prompt_opt
cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
$PY /u/ad11/prompt_opt/slurm/bench_gen.py
