#!/bin/bash
#SBATCH -J whip-direct
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=7G
#SBATCH -t 4:00:00
#SBATCH --array=0-6
#SBATCH -o .slurm-logs/whip-direct-%A_%a.out
#SBATCH --mail-type=END,TIME_LIMIT_80
#SBATCH --mail-user=dsartor97@gmail.com

# the paper's DIRECT-L baseline (deterministic, so one run per target); task 0 is the planar model
cd "$SLURM_SUBMIT_DIR"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then TASK="--planar"; else TASK="--target $SLURM_ARRAY_TASK_ID"; fi
uv run scripts/run_whip_direct.py $TASK --results_dir results/icra --verbose
