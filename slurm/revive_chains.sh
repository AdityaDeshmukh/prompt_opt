#!/bin/bash
###############################################################################
# Sweep the v3 campaign for dead chains and resubmit them.
#
# Two chains died silently and were only noticed days later, both because the
# self-resubmit `sbatch` could not reach the scheduler:
#   task 2  (rrebel_l1_std_seed0) - NODE_FAIL 07-23, found 07-29 (6 days lost)
#   task 10 (rrebel_l1_std_seed2) - slurmctld outage 07-29, found same day
# The EXIT-trap resubmit is best-effort by nature: if the node or controller is
# gone, nothing inside the job can rescue the chain. This sweeper is the
# out-of-band backstop.
#
# A task is considered dead when it is absent from squeue AND its latest
# checkpoint is below MAX_STEPS. Safe to run repeatedly (it no-ops on healthy
# chains) and safe to cron.
#
# Usage: bash slurm/revive_chains.sh          # report only
#        bash slurm/revive_chains.sh --fix    # report and resubmit
###############################################################################
MAX_STEPS=12000
OUT_ROOT=/scratch/ad11/prompt_opt/outputs/v3
STOP_FILE=/u/ad11/prompt_opt/STOP_CAMPAIGN
ALGOS=(grpo rrebel_l1_ent rrebel_l1_std rrebel_huber_std)
CSL_TASKS=" 0 4 8 11 12 13 14 "
MAX_TASK=14   # 0-11 = original matrix, 12-14 = grpo_ent arm (2026-08-02)

run_name() {  # task id -> run directory name (mirrors train_v3.slurm)
    local t=$1
    if [ "$t" -ge 12 ]; then echo "v3_grpo_ent_seed$((t - 12))"
    else echo "v3_${ALGOS[$((t % 4))]}_seed$((t / 4))"; fi
}
FIX=0
[ "$1" = "--fix" ] && FIX=1

if [ -f "$STOP_FILE" ]; then
    echo "STOP_CAMPAIGN present - refusing to revive anything"
    exit 0
fi

# Tasks currently in the queue. Array specs come in several shapes and ALL of
# them must be expanded, or a live chain is misread as dead and "revived" into
# a duplicate -- two chains sharing one checkpoint dir, which is exactly the
# corruption --no-requeue exists to prevent. Handled:
#   9740643_0          single
#   9705999_[0,4,8]    comma list
#   9741220_[12-14]    range      <- missed by the pre-2026-08-02 version
#   9741220_[12-14%2]  range with throttle suffix
live=$(squeue -u ad11 -h -o "%i" \
       | sed -E 's/.*_//; s/[][]//g; s/%.*//' \
       | tr ',' '\n' \
       | awk -F- '/^[0-9]+$/ {print $1}
                  /^[0-9]+-[0-9]+$/ {for (i=$1; i<=$2; i++) print i}' \
       | sort -un | tr '\n' ' ')
echo "live tasks: ${live:-none}"

dead=()
for T in $(seq 0 $MAX_TASK); do
    RUN=$(run_name "$T")
    CKPT_DIR=${OUT_ROOT}/${RUN}/ckpt
    step=$(ls -1v ${CKPT_DIR}/ckpt.step.*.pth 2>/dev/null | tail -1 \
           | sed -E 's/.*ckpt\.step\.([0-9]+)\.pth/\1/')
    step=${step:-0}
    marker=""
    [ -f "${OUT_ROOT}/${RUN}/.chain_died" ] && marker=" [.chain_died]"

    if [[ " $live " == *" $T "* ]]; then
        printf "  task %-2s %-26s alive   step=%s%s\n" "$T" "$RUN" "$step" "$marker"
    elif [ "$step" -ge "$MAX_STEPS" ]; then
        printf "  task %-2s %-26s DONE    step=%s\n" "$T" "$RUN" "$step"
    else
        printf "  task %-2s %-26s DEAD    step=%s%s\n" "$T" "$RUN" "$step" "$marker"
        dead+=("$T")
    fi
done

if [ ${#dead[@]} -eq 0 ]; then
    echo "no dead chains"
    exit 0
fi
echo "dead tasks: ${dead[*]}"
if [ "$FIX" -ne 1 ]; then
    echo "(re-run with --fix to resubmit)"
    exit 1
fi

for T in "${dead[@]}"; do
    RUN=$(run_name "$T")
    PART=secondary
    [[ "$CSL_TASKS" == *" $T "* ]] && PART="secondary,csl"
    if sbatch --partition="$PART" --array=${T} \
              /u/ad11/prompt_opt/slurm/train_v3.slurm; then
        rm -f "${OUT_ROOT}/${RUN}/.chain_died"
        echo "  revived task $T ($RUN) on $PART"
    else
        echo "  FAILED to revive task $T ($RUN) - scheduler still unreachable"
    fi
done
