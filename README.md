## Overview

Functional Bayesian Optimization (FBO) over sparse RKHS manifolds, following the NeurIPS 2026 paper *Bayesian Optimization on Function Spaces via Sparse RKHS Manifolds*.
Candidates are RBF mixtures with per-atom lengthscales; a GP over functions scores them with Monte Carlo qLogEI, and the number of atoms grows in stages.

The code, baselines and sweep that reproduce the paper live at tag [`neurips-2026-camera-ready`](../../tree/neurips-2026-camera-ready).

## Repository Structure

```
.
├── scripts/
│   ├── run.py           # Run a single experiment
│   ├── sweep.sh         # Run all targets and seeds (local pool or SLURM)
│   └── dashboard.py     # Build a standalone HTML dashboard from a results dir
├── src/bofus/
│   ├── kernels.py       # Kernel profiles of distance (RBF, Matérn)
│   ├── rkhs.py          # RBF mixtures and distances between functions
│   ├── gp.py            # GP over functions: fit and conditioning
│   ├── acquisition.py   # qLogEI and its optimization
│   └── utils.py         # Latin hypercube sampling, rescaling, timing
├── src/targets/         # One file per problem class: ridge, projection, gymnasium
└── tests/
```

## Installation

```bash
# Requires Python 3.12+ and uv (https://docs.astral.sh/uv/getting-started/installation/)
uv sync
```

## Running

```bash
uv run scripts/run.py --target_fn gramacylee_ridge --profile matern52 --seed 0 --verbose
```

Targets are `<profile>_ridge` and `<profile>_projection` over the `jaxvlse` profiles (`gramacylee`, `ackley`, `hartmann`, `rosenbrock`, `michalewicz`), and the gymnasium env ids (e.g. `Hopper-v5`) for the closed-loop control tasks.
Ablation: `--fixed_l` shares one lengthscale across atoms.

Results go to `results/neurips/{target_fn}/{method}/{profile}/seed_{seed}_m_{m}.npz` (override with `--results_dir`).

```bash
uv run scripts/dashboard.py --results_dir results/neurips  # writes <results_dir>/dashboard.html
```

## Tests

```bash
uv run pytest
```

## License

MIT, see [LICENSE](LICENSE).
