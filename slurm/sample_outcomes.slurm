#!/bin/bash
# Per-sample outcome audit: 100 test sentences x floors {30,50,70} x 50 task-LM
# samples, prompts decoded as in each campaign's eval (v5: sampled, v4: legacy
# top-3). analysis/example_outputs.py, summarized by analysis/example_table.py.
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=tst_outcomes
#SBATCH --partition=secondary,scavenger
#SBATCH --exclude=ccc0089,ccc0090,ccc0215,ccc0286,ccc0287,ccc0478,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
#SBATCH --prefer=H100|h100|h200|rtx6000b
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err
cd /u/ad11/prompt_opt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/scratch/ad11/hf_cache
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
SENTS="[$(seq -s, 0 5 495)]"
for CAMP in v5 v4; do
  for ARM in rrebel_huber_std rrebel_l1_std; do
    RUN=${CAMP}_${ARM}_seed0; DEC=sample; [ "$CAMP" = v4 ] && DEC=top3
    $PY analysis/example_outputs.py algo=${ARM} random_seed=0 report_to_wandb=false \
        device=cuda:0 device_scorer=cuda:0 task_lm_backend=vllm vllm_seed=0 scorer_dtype=float32 \
        save_dir=/scratch/ad11/prompt_opt/outputs/${CAMP}/${RUN} \
        checkpoint_path=/scratch/ad11/prompt_opt/outputs/${CAMP}/${RUN}/ckpt/ckpt.step.12000.pth \
        "+ex_sents=${SENTS}" "+ex_floors=[30,50,70]" +ex_decode=${DEC} \
        +ex_out=/u/ad11/prompt_opt/results/examples/outcomes_${RUN}.json
  done
done
