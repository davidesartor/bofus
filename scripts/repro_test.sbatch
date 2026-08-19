#!/usr/bin/env bash
#SBATCH --job-name=repro
#SBATCH --output=scripts/repro_%A/%a.out
#SBATCH --array=0-47
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu

set -x
set -e

# one task per (arm, method, profile), seed 0, lengthscale 0.2, sinc1d
COMBOS=()
for ARM in head new; do
for METHOD in random ours vien kundu shilton vellanky; do
for PROFILE in rbf matern52 matern32 matern12; do
    COMBOS+=("$ARM $METHOD $PROFILE")
done
done
done
read -r ARM METHOD PROFILE <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

REPO=$PWD
OUT="$REPO/scripts/repro_out"
mkdir -p "$OUT"

WORK=/tmp/repro_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WORK"
if [[ "$ARM" == head ]]; then
    git -C "$REPO" archive HEAD src pyproject.toml uv.lock | tar -x -C "$WORK"
else
    cp -r "$REPO"/{src,pyproject.toml,uv.lock} "$WORK/"
fi
ln -s "$REPO/.venv" "$WORK/.venv"

cd "$WORK" && uv run --no-sync python -u src/run.py \
    --method=$METHOD --target_fn=sinc1d --profile=$PROFILE --lengthscale=0.2 --seed=0 \
    --initial_acquisitions=10 --minimum_k=1 --maximum_k=10 --acquisitions_each_k=10 \
    --acquisition_raw_samples=1024 --acquisition_max_restarts=16

cp "$WORK/results/sinc1d/$METHOD/${PROFILE}_lengthscale_0.2/seed_0.pkl" \
   "$OUT/${ARM}_${METHOD}_${PROFILE}.pkl"
cd "$REPO" && rm -rf "$WORK"
