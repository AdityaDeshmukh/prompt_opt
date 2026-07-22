#!/bin/bash
###############################################################################
# v3 campaign: GRPO + top-3 R-REBEL variants (from the v2 ranking), MLP
# adaptor, 3 seeds each = 12 single-GPU runs. vLLM task-LM backend
# (prompt_opt_v3 env), outputs on /scratch (home quota killed the v2 chains).
#
# Robust resubmit (fixes v2's silent chain death):
#   - the resubmit decision runs in an EXIT trap, so it fires on clean exit,
#     python crash, AND scancel/preemption TERM;
#   - progress guard with a retry budget: up to 3 consecutive no-progress
#     cycles are retried (a transient crash no longer kills the chain),
#     then the chain gives up loudly;
#   - kill switch: `touch /u/ad11/prompt_opt/STOP_CAMPAIGN` stops all chains
#     at their next cycle end; remove it before relaunching.
#
# Usage: sbatch slurm/train_v3.slurm            (all 12)
#        sbatch --array=3 slurm/train_v3.slurm  (one run)
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_v3
#SBATCH --partition=secondary
#SBATCH --constraint="A100|H100|h100|A40|RTXA6000"
# explicit no-requeue: the EXIT trap owns chain continuation. With requeue,
# preemption would BOTH requeue the job AND let the trap resubmit -> two
# concurrent chains for one run corrupting each other's checkpoints.
#SBATCH --no-requeue
#SBATCH --array=0-11
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%A_%a.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%A_%a.err
#SBATCH --mail-user=ad11@illinois.edu
#SBATCH --mail-type=FAIL

cd /u/ad11/prompt_opt
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0  # native torch sampler; flashinfer needs ninja+nvcc JIT

MAX_STEPS=12000
MAX_NOPROGRESS_RETRIES=3
ALGOS=(grpo rrebel_l1_ent rrebel_l1_std rrebel_huber_std)
T=$SLURM_ARRAY_TASK_ID
ALGO=${ALGOS[$((T % 4))]}
SEED=$((T / 4))
RUN=v3_${ALGO}_seed${SEED}
# GRPO needs a KL anchor to a reference or it entropy-collapses to one
# lambda-independent prompt (KL-free single-update GRPO has no trust region:
# ratio==1 makes the clip inert). 0.04 = DeepSeekMath standard. The rrebel
# family uses `beta`, not grpo_beta, so this arg is a no-op for them.
ALGO_ARGS=""
[ "$ALGO" = "grpo" ] && ALGO_ARGS="grpo_beta=0.04"
# wandb run id (decoupled from the checkpoint dir below). grpo's original ids
# (v3_grpo_seed*) were deleted earlier; resuming a DELETED id hangs
# wandb.init (times out), so grpo logs under a fresh id. Checkpoints still
# resume from RUN_DIR, so training continuity is unaffected.
WANDB_NAME=${RUN}
[ "$ALGO" = "grpo" ] && WANDB_NAME=${RUN}_kl
OUT_ROOT=/scratch/ad11/prompt_opt/outputs/v3
RUN_DIR=${OUT_ROOT}/${RUN}
CKPT_DIR=${RUN_DIR}/ckpt
STOP_FILE=/u/ad11/prompt_opt/STOP_CAMPAIGN
RETRY_FILE=${RUN_DIR}/.noprogress_count
mkdir -p "${RUN_DIR}"

# early kill-switch check, BEFORE the trap is installed: a job that starts
# while STOP_CAMPAIGN exists exits immediately instead of burning a 4h cycle
if [ -f "$STOP_FILE" ]; then
    echo "[$(date)] ${RUN}: STOP_CAMPAIGN present at start - exiting without running"
    exit 0
fi

latest_step() {  # highest checkpoint step, or 0
    local f; f=$(ls -1v ${CKPT_DIR}/ckpt.step.*.pth 2>/dev/null | tail -1)
    [ -z "$f" ] && { echo 0; return; }
    basename "$f" | sed -E 's/ckpt\.step\.([0-9]+)\.pth/\1/'
}

START_STEP=$(latest_step)

resubmit() {  # sbatch with retries: a transient scheduler error must not kill the chain
    local k
    for k in 1 2 3; do
        sbatch --array=${T} /u/ad11/prompt_opt/slurm/train_v3.slurm && return 0
        echo "[$(date)] ${RUN}: sbatch attempt ${k}/3 failed, retrying in 30s"
        sleep 30
    done
    echo "[$(date)] ${RUN}: ALL sbatch attempts failed - chain ends here (investigate!)"
    return 1
}

resubmit_decision() {
    local end_step retries
    end_step=$(latest_step)
    echo "[$(date)] ${RUN}: cycle ended at step ${end_step} (started ${START_STEP})"
    if [ -f "$STOP_FILE" ]; then
        echo "[$(date)] ${RUN}: STOP_CAMPAIGN present - not resubmitting"; return
    fi
    if [ "$end_step" -ge "$MAX_STEPS" ]; then
        echo "[$(date)] ${RUN}: reached ${MAX_STEPS} - done"; return
    fi
    if [ "$end_step" -gt "$START_STEP" ]; then
        rm -f "$RETRY_FILE"
        echo "[$(date)] ${RUN}: progress ${START_STEP}->${end_step}, resubmitting"
        resubmit
        return
    fi
    retries=$(cat "$RETRY_FILE" 2>/dev/null || echo 0)
    retries=$((retries + 1))
    if [ "$retries" -le "$MAX_NOPROGRESS_RETRIES" ]; then
        echo "$retries" > "$RETRY_FILE"
        echo "[$(date)] ${RUN}: NO progress (retry ${retries}/${MAX_NOPROGRESS_RETRIES}), resubmitting anyway"
        resubmit
    else
        echo "[$(date)] ${RUN}: no progress after ${MAX_NOPROGRESS_RETRIES} retries - GIVING UP (investigate!)"
    fi
}
trap resubmit_decision EXIT

CKPT_ARG=""
[ "$START_STEP" -gt 0 ] && CKPT_ARG="checkpoint_path=${CKPT_DIR}/ckpt.step.${START_STEP}.pth"
echo "[$(date)] ${RUN}: resuming from step ${START_STEP} (target ${MAX_STEPS})"

timeout 13200 $PY run_tst_multi_obj.py \
        algo=${ALGO} \
        random_seed=${SEED} \
        device=cuda:0 device_scorer=cuda:0 \
        task_lm_backend=vllm \
        eval_steps=500 \
        run_name=${WANDB_NAME} \
        save_dir=${RUN_DIR} \
        ${ALGO_ARGS} \
        ${CKPT_ARG}
rc=$?
# 124 = timeout ended a still-training cycle: expected, not a failure.
# Leaving it nonzero would mark every normal cycle FAILED and spam
# mail-type=FAIL, drowning out real failures.
[ "$rc" -eq 124 ] && exit 0
exit $rc
