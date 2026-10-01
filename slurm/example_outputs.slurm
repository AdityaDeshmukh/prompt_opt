#!/bin/bash
# Worked input/prompt/output examples for the paper (analysis/example_outputs.py):
# Huber-std under Reward I (v4) and Reward II (v5), same test sentences and floors.
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=tst_examples
#SBATCH --partition=secondary,scavenger
#SBATCH --exclude=ccc0089,ccc0090,ccc0215,ccc0286,ccc0287,ccc0478,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
#SBATCH --prefer=H100|h100|h200|rtx6000b
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err
cd /u/ad11/prompt_opt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/scratch/ad11/hf_cache
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
for CAMP in v5 v4; do
    RUN=${CAMP}_rrebel_huber_std_seed0
    $PY analysis/example_outputs.py algo=rrebel_huber_std random_seed=0 report_to_wandb=false \
        device=cuda:0 device_scorer=cuda:0 task_lm_backend=vllm vllm_seed=0 scorer_dtype=float32 \
        save_dir=/scratch/ad11/prompt_opt/outputs/${CAMP}/${RUN} \
        checkpoint_path=/scratch/ad11/prompt_opt/outputs/${CAMP}/${RUN}/ckpt/ckpt.step.12000.pth \
        "+ex_sents=[403,438,352]" "+ex_floors=[0,30,50,70,90]" \
        +ex_out=/u/ad11/prompt_opt/results/examples/${RUN}.json
done
