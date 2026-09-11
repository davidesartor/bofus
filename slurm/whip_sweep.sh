#!/bin/bash
#SBATCH -J whip-bo
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=7G
#SBATCH -t 12:00:00
#SBATCH --array=0-167
#SBATCH -o .slurm-logs/whip-bo-%A_%a.out
#SBATCH --mail-type=END,TIME_LIMIT_80
#SBATCH --mail-user=dsartor97@gmail.com

# 7 tasks (planar + 6 spatial targets) x 3 fixed basis sizes x 8 seeds, 4 initial points and 256 acquisitions
cd "$SLURM_SUBMIT_DIR"
T=$((SLURM_ARRAY_TASK_ID % 7))
if [ "$T" -eq 0 ]; then TASK=whip2d; else TASK=whip3d_$T; fi
SEED=$((SLURM_ARRAY_TASK_ID / 21))
case $((SLURM_ARRAY_TASK_ID / 7 % 3)) in
  0) M=1 ;;
  1) M=2 ;;
  2) M=4 ;;
esac
uv run scripts/run.py --target_fn $TASK --profile matern52 --seed $SEED \
  --m $M --initial_acquisitions 4 --budget 256 --fixed_m --log_objective --results_dir results/icra --verbose
