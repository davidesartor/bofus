#!/usr/bin/env bash
#SBATCH --job-name=plots
#SBATCH --output=logs/plots_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=cpu

cd /home/dsartor_umass_edu/BOFUS
uv run python -u -m bofus.plots
