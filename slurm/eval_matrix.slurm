#!/bin/bash
###############################################################################
# Final test-set evaluation (lambda sweep 0..0.9 on the 500 test sources)
# for every finished run under outputs/v2. Usage: sbatch slurm/eval_matrix.slurm
###############################################################################
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=tst_v2_eval
#SBATCH --partition=csl
#SBATCH --output=/u/ad11/prompt_opt/logs/%x_%j.out
#SBATCH --error=/u/ad11/prompt_opt/logs/%x_%j.err
#SBATCH --mail-user=ad11@illinois.edu
#SBATCH --mail-type=END,FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate prompt_opt
cd /u/ad11/prompt_opt
# call the env python directly; `conda activate` does not switch
# the interpreter in a non-interactive SLURM shell here
PY=/u/ad11/miniconda3/envs/prompt_opt/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for RUN_DIR in /u/ad11/prompt_opt/outputs/v2/*/; do
    RUN=$(basename ${RUN_DIR})
    CKPT=$(ls -1v ${RUN_DIR}/ckpt/ckpt.step.*.pth 2>/dev/null | tail -1)
    [ -z "$CKPT" ] && continue
    [ -f "${RUN_DIR}/test/output.json" ] && continue
    ALGO=$(echo ${RUN} | sed -E 's/^v2(lora)?_//; s/_seed[0-9]+$//')
    EXTRA=""
    [[ ${RUN} == v2lora_* ]] && EXTRA="adaptor_type=hyperlora"
    $PY run_eval.py \
            algo=${ALGO} ${EXTRA} \
            report_to_wandb=false \
            device=cuda:0 device_scorer=cuda:0 \
            scorer_dtype=float32 \
            save_dir=${RUN_DIR} \
            checkpoint_path=${CKPT}
done
