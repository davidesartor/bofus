#!/bin/bash
#SBATCH -J whip-bo
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=7G
#SBATCH -t 12:00:00
#SBATCH --array=0-17
#SBATCH -o .slurm-logs/whip-bo-%A_%a.out
#SBATCH --mail-type=END,TIME_LIMIT_80
#SBATCH --mail-user=dsartor97@gmail.com

# 6 targets x 3 basis sizes; the initial design is sized so every run ends at 128 acquisitions
cd "$SLURM_SUBMIT_DIR"
TARGET=$((SLURM_ARRAY_TASK_ID % 6 + 1))
case $((SLURM_ARRAY_TASK_ID / 6)) in
  0) M=1; INIT=64 ;;
  1) M=2; INIT=32 ;;
  2) M=4; INIT=16 ;;
esac
uv run scripts/run.py --target_fn whip$TARGET --profile matern52 --seed 0 \
  --m $M --initial_acquisitions $INIT --verbose
