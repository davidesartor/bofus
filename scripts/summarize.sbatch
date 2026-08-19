#!/usr/bin/env bash
#SBATCH --job-name=summarize
#SBATCH --output=logs/summarize_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=cpu

REPO=/home/dsartor_umass_edu/BOFUS
WORKTREE=/home/dsartor_umass_edu/BOFUS-pickle

# the sweep keeps writing while the conversion runs, so sweep up whatever
# landed after each shard took its file listing
cd "$WORKTREE"
VIRTUAL_ENV="$REPO/.venv" PYTHONPATH="$WORKTREE" \
    uv run --no-project python -u scripts/convert_pickles.py \
    --source "$REPO/results" --dest "$REPO/results_converted"

cd "$REPO"
uv run python -u -m bofus.summary
