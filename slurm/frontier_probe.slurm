#!/bin/bash
# Tradeoff-curve geometry probe (analysis/frontier_probe.py): policy map on a
# fine lambda grid + per-sample cross-evaluation of the shared prompt menu.
# One array task per arm; task k also scores menu chunk k.
#   sbatch --array=0-3 slurm/frontier_probe.slurm
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=tst_v4_fprobe
#SBATCH --partition=secondary,scavenger
#SBATCH --exclude=ccc0089,ccc0090,ccc0215,ccc0286,ccc0287,ccc0478,ccc0232,ccc0233,ccc0234,ccc0235,ccc0236
#SBATCH --prefer=H100|h100|h200|rtx6000b
#SBATCH --requeue
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%A_%a.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%A_%a.err
cd /u/ad11/prompt_opt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/scratch/ad11/hf_cache
ARMS=(rrebel_l1_std rrebel_huber_std grpo_ent grpo_baseref)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
ALGO=$ARM; ALGO_ARGS=""
# must match training: ScoreLossModule is built from these even for eval
if [ "$ARM" = "grpo_ent" ]; then
    ALGO=grpo; ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.01"
elif [ "$ARM" = "grpo_baseref" ]; then
    ALGO=grpo; ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.0 grpo_ref_mode=base_lm"
fi
RUN=v4_${ARM}_seed0
RUN_DIR=/scratch/ad11/prompt_opt/outputs/v4/${RUN}
CKPT=${RUN_DIR}/ckpt/ckpt.step.12000.pth
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
echo "[$(date)] ${RUN} node=${SLURMD_NODENAME} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python analysis/frontier_probe.py \
    algo=${ALGO} ${ALGO_ARGS} random_seed=0 report_to_wandb=false \
    device=cuda:0 device_scorer=cuda:0 task_lm_backend=vllm vllm_seed=0 \
    scorer_dtype=float32 save_dir=${RUN_DIR} checkpoint_path=${CKPT} \
    +probe_arm=${ARM} +probe_out=/u/ad11/prompt_opt/results/v4/frontier_probe
