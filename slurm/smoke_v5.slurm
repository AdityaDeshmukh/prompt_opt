#!/bin/bash
# 6-step smoke test of the v5 reward + sampled eval through run_tst_multi_obj.py,
# for one R-REBEL arm and one GRPO arm. Writes to a throwaway scratch dir.
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=tst_v5_smoke
#SBATCH --partition=secondary,scavenger
#SBATCH --exclude=ccc0089,ccc0090,ccc0215,ccc0286,ccc0287,ccc0478,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
#SBATCH --prefer=H100|h100|h200|rtx6000b
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err
cd /u/ad11/prompt_opt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/scratch/ad11/hf_cache
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
SMOKE=/scratch/ad11/prompt_opt/outputs/v5_smoke
rm -rf $SMOKE; mkdir -p $SMOKE
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
for spec in "rrebel_huber_std|" "grpo|grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.0 grpo_ref_mode=base_lm"; do
    ALGO=${spec%%|*}; ARGS=${spec#*|}
    echo "===== $ALGO $ARGS"
    $PY run_tst_multi_obj.py algo=$ALGO $ARGS random_seed=0 device=cuda:0 device_scorer=cuda:0 \
        task_lm_backend=vllm report_to_wandb=false max_train_steps=6 eval_steps=5 save_steps=0 \
        print_every=1 save_dir=$SMOKE/$ALGO \
        reward_constraint=expectation constraint_kappa=5.0 eval_decode=sample || echo "FAILED $ALGO rc=$?"
    ls -R $SMOKE/$ALGO | head -20
done
