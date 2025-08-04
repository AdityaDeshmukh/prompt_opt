#!/bin/bash

###############################################################################
#
#SBATCH --gres=gpu:L40S:2                  # GPU
#SBATCH --time=7-00:00:00                # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=32             # Number of task (cores/ppn) per node
#SBATCH --job-name=grpo              # Name of batch job
#SBATCH --partition=csl                  # Partition (queue)
#SBATCH --output=grpo.output%j             # Name of batch job output file
#SBATCH --error=grpo.error%j              # Name of batch job error file
#SBATCH --mail-user=ad11@illinois.edu    # Send email notifications
#SBATCH --mail-type=BEGIN,END            # Type of email notifications to send
#
###############################################################################

conda activate base
conda activate prompt_opt
python run_tst_multi_obj.py \
        algo=grpo \
        train_batch_size=15 \
        num_repeats=16 \
        num_samples=50 \
        save_dir=/u/ad11/prompt_opt/grpo \
        checkpoint_path=/u/ad11/prompt_opt/grpo/ckpt/ckpt.step.3900.pth \

# python run_tst_mo.py \
#         dataset=yelp \
#         direction=0_to_1 \
#         train_batch_size=15 \
#         num_repeats=15 \
#         eval_steps=5 \
#         update_steps=150 \
#         save_steps=150 \
#         num_samples=15 \
#         num_bootstraps=1 \
#         learning_rate=0.0001 \
#         save_dir=./outputs2 \
#         1> output2.out 2> output2.err &

# wait
