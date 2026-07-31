## Overview

Code for submission (NeurIPS 2026) for reproducibility.
We propose a method for Functional Bayesian Optimization (FBO) that searches a sparse sub-manifold of an RKHS by jointly optimizing over basis point locations and coefficients. The repository contains implementations of Ours and four baselines (Vien, Kundu, Shilton, Vellanki), along with the ~50 BO benchmark functions of the [`jaxvlse`](https://pypi.org/project/jaxvlse/) package extended to the functional domain.

## Repository Structure

```
.
├── run.py               # Main entry point: run a single experiment
├── sweep.sh             # SLURM sweep script: launches all experiments
├── summary.py           # Aggregate raw results into parquet tables
├── plots.py             # Generate all figures and tables from the parquet tables
├── src/
│   ├── gp.py            # Gaussian process surrogate models
│   ├── rkhs.py          # RKHS function representations
│   ├── kernels.py       # Kernel functions (RBF, Matérn)
│   ├── acquisition.py   # Acquisition functions and optimization
│   └── targets/         # Functional benchmarks (ridge, sinc) and real-world tasks
├── results/             # Experiment outputs (see below)
└── plots/               # Generated figures
```

## Installation

```bash
# Requires Python 3.12+. We use uv for dependency management.
pip install uv
uv sync
```

## Pre-computed Results

We provide an archive of all pre-computed results — no need to rerun the sweep to reproduce the figures. Download and extract:

```bash
tar -xzf results.tar.gz
```

The archive contains:
- `results/{target_fn}/{method_dir}/{profile}_lengthscale_{lengthscale}/seed_{seed}.pkl` — raw per-run results
- `results/summary_all.parquet` — aggregated summary across all hyperparameter combinations
- `results/summary_filtered.parquet` — summary filtered to the best profile and lengthscale per (method, target)
- `results/ys_all.parquet` — per-acquisition observation values for all runs
- `results/ys_filtered.parquet` — same, filtered to the best hyperparameter combination

`method_dir` is the method name plus a suffix per ablation flag (e.g. `ours_no_natural_grad`, `ours_rcond_1e-08`, `ours_fixed_k_5`, `shilton_reduced_grid`, `ours_constrain_order`).

To regenerate the tables from the raw `.pkl` files (e.g. after running new experiments):

```bash
uv run summary.py
# or rescan only the configs written in the last 15 minutes
uv run summary.py -u 15m
```

## Reproducing Figures

With the results archive extracted and the tables in place:

```bash
uv run plots.py
```

This writes all figures to `plots/`, organized as:

| Directory | Contents |
|---|---|
| `plots/method_comparison/` | Figure 1: running best per target, all methods |
| `plots/tables/` | Tables: average and final regret, fit/acquisition timings, best hyperparameters, per-lengthscale ablation |
| `plots/timings/` | Fit and acquisition time box plots per target |
| `plots/f_visualizations/` | MNIST learned activation, brachistochrone path, hopper animation |
| `plots/natural_gradient_ablation_ours/` | Appendix: natural gradient ablation for ours |
| `plots/natural_gradient_ablation_vien/` | Appendix: natural gradient ablation for Vien |
| `plots/preconditioner_regularization_ablation/` | Appendix: singular value truncation of the preconditioner |
| `plots/constrained_search_ablation/` | Appendix: ordered/simplex-constrained basis point search |
| `plots/candidates_sampling_ablation/` | Appendix: initial candidate distribution |
| `plots/shilton_reduced_grid_ablation/` | Appendix: full vs reduced grid for Shilton's method |
| `plots/fixed_k_ablation/` | Appendix: growing basis size vs fixed k |
| `plots/kernel_profile_ablation/` | Appendix: GP kernel profile sweep |
| `plots/convergence/` | Appendix: Monte Carlo convergence of the ridge functionals |

`RESULTS_DIR` and `PLOTS_DIR` override the input and output directories of both scripts.

## Running Individual Experiments to Spot Check

A single run takes a method, target function, and hyperparameters:

```bash
uv run run.py \
    --method ours \
    --target_fn brachistochrone \
    --profile matern52 \
    --lengthscale 0.2 \
    --seed 0
```

**Methods:** `ours`, `vien`, `kundu`, `shilton`, `vellanky`, `random`

**Target functions:** `sinc1d`, `sinc2d`, `sinc3d`, `sinc4d`, `gramacylee`, `ackley`, `hartmann`, `rosenbrock`, `michalewicz`, `brachistochrone`, `pendulum`, `pinwheel`, `hopper`, `mnist`

**Simulation parameters:** `--initial_acquisitions` (10), `--minimum_k` (1), `--maximum_k` (10), `--acquisitions_each_k` (10), `--acquisition_raw_samples` (1024), `--acquisition_max_restarts` (16)

**Ablation flags:**
- `--disable_natural_gradient` — use Euclidean gradient instead of natural gradient
- `--lstsq_rcond` — truncate preconditioner directions whose singular value falls below this fraction of the largest (default: NumPy's machine-precision cutoff)
- `--sample_candidates_from_gp` — sample LHS candidates from the GP prior (ours only)
- `--reduced_grid` — use a reduced grid for Shilton's method
- `--constrain_order` — constrain the basis points to an ordered simplex (ours only)
- `--fixed_k` — hold the basis size at this k, spending the whole evaluation budget there

Results are saved to `results/{target_fn}/{method_dir}/{profile}_lengthscale_{lengthscale}/seed_{seed}.pkl`.

## Running the Full Sweep (SLURM)

The sweep script replicates all experiments in the paper. It submits SLURM array jobs per target function and reruns any missing results every 10 minutes until a deadline:

```bash
bash sweep.sh main 72   # watcher named "main", running for up to 72 hours (default)
```

The first argument names the watcher: watchers under different names keep their own log (`sweep_{name}.log`) and manifests, so two of them never hand out the same run.

The full sweep covers 18 method variants × 4 profiles × 4 lengthscales × 16 seeds across 14 target functions, minus the variants that do not apply to every target (Vellanki's parametrization is 1d only; the constrained-search ablation is defined for the ridge and physical targets). Each run is one array task on one cpu, with a per-target wall clock from 30 minutes (sinc projections) to 6 hours (MNIST). The script skips runs whose output file already exists, so a preempted or timed-out task is picked up on a later pass.

After the sweep completes, regenerate the tables and plots:

```bash
uv run summary.py
uv run plots.py
```

## Benchmark Functions

The scalar test functions come from [`jaxvlse`](https://pypi.org/project/jaxvlse/), a JAX port of the [virtual library of simulation experiments](http://www.sfu.ca/~ssurjano) (Surjanovic & Bingham 2013) that supports `jit` and `grad`. `Ridge` extends any of them to the functional domain via the ridge function construction described in Section 5 of the paper, taking the input dimension from the profile.

```python
import vlse

from src.targets import Ridge

# Hartmann3 as a functional benchmark (d=3 dimensional input)
target = Ridge(vlse.Hartmann3(normalized=True))
```

`normalized=True` puts the profile on the unit hypercube with its global minimum at 0, which is what `Ridge` feeds it.
