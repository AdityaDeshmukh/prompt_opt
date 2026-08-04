#!/bin/bash
# Waits until every v3 chain has finished (or died), then prints the final
# matched-step table. The scientific question is already settled -- both grpo
# arms plateau while R-REBEL climbs -- so this just fills in the 12000-step
# numbers for the paper table.
PY=/u/ad11/miniconda3/envs/prompt_opt_v3/bin/python
CAP_HOURS=${1:-72}
DEADLINE=$(( $(date +%s) + CAP_HOURS*3600 ))

echo "[$(date)] waiting for all v3 chains to finish (${CAP_HOURS}h cap)"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    census=$(bash /u/ad11/prompt_opt/slurm/revive_chains.sh)
    alive=$(echo "$census" | grep -c 'alive')
    dead=$(echo "$census" | grep -c 'DEAD')
    if [ "$alive" -eq 0 ]; then
        echo "[$(date)] all chains finished (dead=${dead})"
        echo "$census" | tail -17
        echo
        cd /u/ad11/prompt_opt && $PY analysis/final_table.py
        exit 0
    fi
    echo "[$(date)] alive=${alive} dead=${dead} | $(echo "$census" \
         | grep alive | sed -E 's/.*(v3_[a-z0-9_]+) +alive +step=([0-9]+)/\1:\2/' \
         | tr '\n' ' ')"
    sleep 1800
done
echo "[$(date)] ${CAP_HOURS}h cap reached; chains still running"
bash /u/ad11/prompt_opt/slurm/revive_chains.sh | tail -17
exit 2
