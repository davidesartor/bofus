#!/bin/bash
#SBATCH -J whip-sparsity
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o .slurm-logs/whip-sparsity-%j.out
#SBATCH --mail-type=END,TIME_LIMIT_80
#SBATCH --mail-user=dsartor97@gmail.com

# sparsity of the paper's min-jerk solutions: RBF mixture fits to the 3D and planar references
cd "$SLURM_SUBMIT_DIR"
uv run scripts/fit_paper_references.py
