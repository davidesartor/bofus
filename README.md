## Overview

Functional Bayesian Optimization (FBO) over sparse RKHS manifolds, following the NeurIPS 2026 paper *Bayesian Optimization on Function Spaces via Sparse RKHS Manifolds*.
Candidates are RBF mixtures with per-atom lengthscales; a GP over functions scores them with Monte Carlo qLogEI, and the number of atoms grows in stages.

The code, baselines and sweep that reproduce the paper live at tag [`neurips-2026-camera-ready`](../../tree/neurips-2026-camera-ready).

## Repository Structure

```
.
├── scripts/
│   ├── run.py           # Run a single experiment
│   ├── make_viz.py      # Build a standalone HTML visualization from a run
│   ├── run_whip_direct.py         # DIRECT-L baseline on the whip, ported from Nah et al.
│   ├── fit_paper_references.py    # Fit RBF mixtures to the paper's min-jerk moves
│   ├── make_whip_dashboard.py     # Dashboard over the whip runs, every seed
│   ├── make_sparsity_dashboard.py # Dashboard of the m-bump reference fits
│   └── templates/       # HTML templates filled by the scripts
├── slurm/               # Whip sweep, DIRECT-L and reference-fit jobs
├── src/bofus/
│   ├── kernels.py       # Kernel profiles of distance (RBF, Matérn)
│   ├── rkhs.py          # RBF mixtures
│   ├── gp.py            # GP over functions
│   ├── acquisition.py   # qLogEI and its optimization
│   └── utils.py         # Latin hypercube sampling, rescaling, timing
├── src/targets/         # Functional benchmarks (ridge, projection) and real-world tasks
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
uv run scripts/make_viz.py <run.npz>    # writes plots/{target}_{npz stem}.html
```

Results go to `{results_dir}/{target_fn}/{method}/{profile}/seed_{seed}_m_{m}.npz`.

## Tests

```bash
uv run pytest
```

## License

MIT, see [LICENSE](LICENSE).
