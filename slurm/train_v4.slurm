#!/bin/bash
###############################################################################
# v4 frontier regeneration: 3 arms x seed 0 = 3 single-GPU runs, 12000 steps.
#
# WHY THIS EXISTS
# The v3 campaign completed (all 15 runs, 12000 steps) but /scratch was purged
# around 2026-09-03..05: every ckpt/ and eval/ under outputs/v3/ is now empty
# (0 *.pth, 0 outputs.step.*.json anywhere under /scratch/ad11). The scalar
# results survive on wandb, but score_trainer.evaluate() collapses the lambda
# sweep before logging (mean over lmbdas), so wandb has only lambda-AVERAGED
# content/style. The per-lambda arrays needed for the content-vs-sentiment
# tradeoff curve existed ONLY in the purged eval JSONs. This rerun regenerates
# them at one seed per arm.
#
# DELIBERATELY IDENTICAL to train_v3.slurm: algo args, eval_steps, max steps,
# env, partition strategy, group size, batch sizes. Do not "improve" anything
# here -- the point is that v4 seed0 be comparable to v3 seed0.
#
# DELIBERATELY DIFFERENT:
#   - run/wandb ids are v4_* and outputs go to outputs/v4/. Resuming the v3
#     wandb ids would append new history to the FINISHED v3 runs and destroy
#     the only surviving record of the completed campaign.
#   - archive_cycle(): after every cycle, eval JSONs (+ the newest checkpoint)
#     are copied to /u/ad11/prompt_opt/results/v4/ in HOME. This is the fix for
#     the root cause of the v3 data loss -- scratch is purgeable, so nothing
#     that a paper depends on may live only there.
#
# Usage: sbatch slurm/train_v4.slurm             (all 3)
#        sbatch --array=2 slurm/train_v4.slurm   (one run)
#        touch /u/ad11/prompt_opt/STOP_V4        (stop all chains at cycle end)
###############################################################################
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_v4
#SBATCH --partition=secondary
# node-select via --exclude (see train_v3.slurm): drop V100/T4 (emulated bf16,
# ~12x slower -> a cycle never reaches the 150-step checkpoint interval and the
# chain dies silently) and A30 (24GB, too small for max_gen_batch_size=400).
#SBATCH --exclude=ccc0215,ccc0286,ccc0287,ccc0478
# the EXIT trap owns chain continuation; requeue would create a second
# concurrent chain for the same run and corrupt its checkpoints.
#SBATCH --no-requeue
#SBATCH --array=0-2
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
SEED=0
# 3 arms (trimmed from 5 on 2026-09-08 so the array schedules -- both partitions
# were saturated and a 5-GPU array was estimated >1 day out):
#   rrebel_l1_std  the best R-REBEL variant (tied with huber_std within noise)
#   grpo_ent       the DEFENSIBLE GRPO baseline. Plain `grpo` is dropped because
#                  the audit showed it never learns: arm mean 30.38 @500 ->
#                  30.19 @11500, KL pinned at the k3 ceiling, median 5 distinct
#                  tokens per cycle = one frozen 5-token prompt. See AUDIT.md.
#   grpo_baseref   NEW, and the experiment the audit says was missing. Plain
#                  GRPO but with grpo_ref_mode=base_lm, so the pinned KL anchor
#                  is the unadapted backbone LM instead of the initial policy.
#                  The v3 anchor was numerically UNIFORM (measured entropy
#                  10.824907 vs ln(50257)=10.824905), which makes beta*KL an
#                  entropy bonus rather than a trust region; base_lm measures
#                  7.672 nats, so the anchor finally exerts real force. This is
#                  the arm that answers "was your baseline actually GRPO?".
ARMS=(rrebel_l1_std grpo_ent grpo_baseref)
T=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$T]}
ALGO=$ARM
ALGO_ARGS=""
if [ "$ARM" = "grpo_ent" ]; then
    ALGO=grpo
    # 0.04 = DeepSeekMath standard. grpo_ref_sync_steps=0 pins the anchor for
    # the whole run (TRL/verl/OpenRLHF behaviour); the v3 bug was re-deepcopying
    # it every update_steps=150, which zeroes the KL and its gradient at each
    # refresh so the anchor exerts no restoring force.
    ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.01"
elif [ "$ARM" = "grpo_baseref" ]; then
    ALGO=grpo
    ALGO_ARGS="grpo_beta=0.04 grpo_ref_sync_steps=0 grpo_ent_coef=0.0 grpo_ref_mode=base_lm"
fi
RUN=v4_${ARM}_seed${SEED}
WANDB_NAME=${RUN}
# csl is physically 2 nodes, so at most 2 of our jobs can ever sit there. With
# only 3 arms all of them are eligible for both partitions; note this PART only
# takes effect on RESUBMISSION -- the initial sbatch uses the #SBATCH default,
# so pass --partition on the command line for the first submission.
CSL_TASKS=" 0 1 2 "  # all 3 arms may take csl or secondary, whichever frees first
PART=secondary
[[ "$CSL_TASKS" == *" $T "* ]] && PART="secondary,csl"

OUT_ROOT=/scratch/ad11/prompt_opt/outputs/v4
RUN_DIR=${OUT_ROOT}/${RUN}
CKPT_DIR=${RUN_DIR}/ckpt
EVAL_DIR=${RUN_DIR}/eval
# HOME archive: purge-proof copy of everything the paper depends on.
ARCHIVE_DIR=/u/ad11/prompt_opt/results/v4/${RUN}
STOP_FILE=/u/ad11/prompt_opt/STOP_V4
RETRY_FILE=${RUN_DIR}/.noprogress_count
mkdir -p "${RUN_DIR}" "${ARCHIVE_DIR}/eval"

if [ -f "$STOP_FILE" ]; then
    echo "[$(date)] ${RUN}: STOP_V4 present at start - exiting without running"
    exit 0
fi

latest_step() {  # highest checkpoint step, or 0
    local f; f=$(ls -1v ${CKPT_DIR}/ckpt.step.*.pth 2>/dev/null | tail -1)
    [ -z "$f" ] && { echo 0; return; }
    basename "$f" | sed -E 's/ckpt\.step\.([0-9]+)\.pth/\1/'
}

archive_cycle() {
    # Copy the per-lambda eval JSONs (small, and the ONLY source of the
    # tradeoff curve) plus the newest checkpoint into home. Runs every cycle,
    # not just at the end, so a purge or a dead chain can never again cost us
    # the whole campaign. cp -u = only changed/new files.
    local n=0 newest
    if [ -d "$EVAL_DIR" ]; then
        cp -u ${EVAL_DIR}/outputs.step.*.json "${ARCHIVE_DIR}/eval/" 2>/dev/null
        n=$(ls -1 ${ARCHIVE_DIR}/eval/outputs.step.*.json 2>/dev/null | wc -l)
    fi
    newest=$(ls -1v ${CKPT_DIR}/ckpt.step.*.pth 2>/dev/null | tail -1)
    if [ -n "$newest" ]; then
        cp -u "$newest" "${ARCHIVE_DIR}/" 2>/dev/null
        # keep only the newest archived checkpoint (home quota is 100G)
        ls -1v ${ARCHIVE_DIR}/ckpt.step.*.pth 2>/dev/null | head -n -1 | xargs -r rm -f
    fi
    echo "[$(date)] ${RUN}: archived ${n} eval JSONs + $(basename "${newest:-none}") -> ${ARCHIVE_DIR}"
}

START_STEP=$(latest_step)

resubmit() {  # sbatch with time-bounded retries (rides out a slurmctld outage)
    local deadline=$(( $(date +%s) + 900 ))   # 15 min
    local wait=30 k=0
    while [ "$(date +%s)" -lt "$deadline" ]; do
        k=$((k + 1))
        sbatch --partition="$PART" --array=${T} /u/ad11/prompt_opt/slurm/train_v4.slurm && return 0
        echo "[$(date)] ${RUN}: sbatch attempt ${k} failed, retrying in ${wait}s"
        sleep "$wait"
        wait=$(( wait * 2 )); [ "$wait" -gt 120 ] && wait=120
    done
    echo "[$(date)] ${RUN}: ALL sbatch attempts failed over 15min - chain ends here"
    date +"%Y-%m-%dT%H:%M:%S%z chain died: sbatch unreachable for 15min" \
        > "${RUN_DIR}/.chain_died"
    return 1
}

resubmit_decision() {
    local end_step retries
    end_step=$(latest_step)
    archive_cycle
    echo "[$(date)] ${RUN}: cycle ended at step ${end_step} (started ${START_STEP})"
    if [ -f "$STOP_FILE" ]; then
        echo "[$(date)] ${RUN}: STOP_V4 present - not resubmitting"; return
    fi
    if [ "$end_step" -ge "$MAX_STEPS" ]; then
        echo "[$(date)] ${RUN}: reached ${MAX_STEPS} - done"
        # hand off to the final fixed-seed per-lambda eval
        sbatch --array=${T} /u/ad11/prompt_opt/slurm/eval_v4.slurm \
            && echo "[$(date)] ${RUN}: submitted final eval" \
            || echo "[$(date)] ${RUN}: final eval sbatch FAILED - run eval_v4.slurm by hand"
        return
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
echo "[$(date)] ${RUN}: arm=${ARM} algo=${ALGO} args='${ALGO_ARGS}' resuming from step ${START_STEP} (target ${MAX_STEPS})"

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
# 124 = `timeout` ended a still-training cycle: expected, not a failure.
[ "$rc" -eq 124 ] && exit 0
exit $rc
