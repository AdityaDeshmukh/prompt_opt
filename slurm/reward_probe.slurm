#!/bin/bash
# Reward-scale probe for the R-REBEL-std theory (analysis/reward_scale_probe.py).
# One array task per checkpoint step of the rrebel_l1_std arm.
#   sbatch --array=0-2 slurm/reward_probe.slurm
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=tst_v4_probe
#SBATCH --partition=secondary,scavenger
#SBATCH --exclude=ccc0089,ccc0090,ccc0215,ccc0286,ccc0287,ccc0478,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
#SBATCH --prefer=H100|h100|h200|rtx6000b
#SBATCH --requeue
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%A_%a.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%A_%a.err
cd /u/ad11/prompt_opt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/scratch/ad11/hf_cache
STEPS=(150 3000 12000)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
RUN=v4_rrebel_l1_std_seed0
CKPT=/scratch/ad11/prompt_opt/outputs/v4/${RUN}/ckpt/ckpt.step.${STEP}.pth
/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python analysis/reward_scale_probe.py \
    algo=rrebel_l1_std random_seed=0 report_to_wandb=false \
    device=cuda:0 device_scorer=cuda:0 task_lm_backend=vllm vllm_seed=0 \
    scorer_dtype=float32 max_size=40 eval_batch_size=40 \
    "eval_lmbdas=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]" \
    save_dir=/scratch/ad11/prompt_opt/outputs/v4/${RUN} checkpoint_path=${CKPT} \
    +probe_out=/u/ad11/prompt_opt/results/v4/reward_scale_probe/${RUN}.step${STEP}.json
