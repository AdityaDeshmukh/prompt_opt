#!/bin/bash
# Waits for the first step-500 eval from the fixed-anchor GRPO relaunch, then
# runs the collapse gate. Exits as soon as there is an actionable signal, so
# an early failure is caught without waiting for all three seeds.
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
V3=/scratch/ad11/prompt_opt/outputs/v3
DEADLINE=$(( $(date +%s) + 24*3600 ))

echo "[$(date)] watching for step-500 evals from v3_grpo_seed{0,1,2} (24h cap)"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    n=0
    for s in 0 1 2; do
        [ -f "${V3}/v3_grpo_seed${s}/eval/outputs.step.500.json" ] && n=$((n+1))
    done
    if [ "$n" -gt 0 ]; then
        echo "[$(date)] ${n}/3 seed(s) reached step 500 - running gate"
        cd /u/ad11/prompt_opt && $PY analysis/gate_grpo_500.py 500
        rc=$?
        echo "[$(date)] gate exit=${rc} (0=pass, 1=collapse/escalate)"
        # also show whether the KL is actually binding now
        echo "--- newest grpo cycle log (KL/anchor sanity) ---"
        lg=$(ls -t /u/ad11/prompt_opt/logs/tst_v3_*_[048].out 2>/dev/null | head -1)
        [ -n "$lg" ] && grep -E "NOTE: algo=grpo|ref_sync_steps" "$lg" | head -2
        exit $rc
    fi
    # progress heartbeat so a stalled queue is visible
    running=$(squeue -u ad11 -h -o "%i %T" | grep -c RUNNING)
    echo "[$(date)] no step-500 eval yet | jobs running=${running}"
    sleep 600
done
echo "[$(date)] 24h cap reached with no step-500 eval - investigate queue"
exit 2
