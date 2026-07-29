#!/bin/bash
# Waits until a GRPO seed has enough eval points to judge the TREND, then runs
# the health gate. Step 500 alone was not decisive: the fixed anchor scored
# 34.1 there (above R-REBEL's 30.1-33.1 band) but with only 6/100 distinct
# prompts, so the open question is whether it keeps climbing like R-REBEL
# (33 -> 40 by step 6000) or plateaus around 34.
TARGET=${1:-1500}          # need evals at 500/1000/1500 for a 3-point trend
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
V3=/scratch/ad11/prompt_opt/outputs/v3
CAP_HOURS=${2:-36}
DEADLINE=$(( $(date +%s) + CAP_HOURS*3600 ))

echo "[$(date)] waiting for a GRPO seed to reach step ${TARGET} (${CAP_HOURS}h cap)"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    hit=""
    for s in 0 1 2; do
        [ -f "${V3}/v3_grpo_seed${s}/eval/outputs.step.${TARGET}.json" ] \
            && hit="${hit}${s} "
    done
    if [ -n "$hit" ]; then
        echo "[$(date)] seed(s) ${hit}reached step ${TARGET} - running gate"
        cd /u/ad11/prompt_opt && $PY analysis/gate_grpo.py "$TARGET"
        rc=$?
        echo "[$(date)] gate exit=${rc} (0=ok, 1=pathological, 2=no data)"
        echo "--- chain census ---"
        bash /u/ad11/prompt_opt/slurm/revive_chains.sh | tail -14
        exit $rc
    fi
    steps=""
    for s in 0 1 2; do
        last=$(ls "${V3}/v3_grpo_seed${s}/eval" 2>/dev/null \
               | sed 's/[^0-9]*//g' | sort -n | tail -1)
        steps="${steps}s${s}=${last:-0} "
    done
    echo "[$(date)] ${steps}| $(squeue -u ad11 -h -o '%T' 2>/dev/null \
          | grep -c RUNNING) running"
    sleep 900
done
echo "[$(date)] ${CAP_HOURS}h cap reached without hitting step ${TARGET}"
exit 2
