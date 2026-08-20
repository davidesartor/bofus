from collections import defaultdict
from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
import equinox as eqx
import scipy as sp
import numpy as np

import argparse
import time
import os
import pickle

from bofus import gp, kernels, acquisition, rkhs
import targets

jax.config.update("jax_enable_x64", True)

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/neurips")

VERBOSE = False

SAVE_DTYPE = np.float32


def downcast(tree, dtype: np.dtype = SAVE_DTYPE):
    """Store every float array at reduced precision; the pickles run to gigabytes otherwise."""

    def cast(leaf):
        if isinstance(leaf, (jax.Array, np.ndarray)) and leaf.dtype.kind == "f":
            return np.asarray(leaf, dtype)
        return leaf

    return jax.tree.map(cast, tree)


def vprint(*args, **kwargs) -> None:
    """Print only under --verbose; sweeps run thousands of these and the logs add up."""
    if VERBOSE:
        print(*args, **kwargs)


# the acquisition loss lives at module level, so every varying quantity is an argument and
# the jit cache is reused across iterations instead of being rebuilt with each closure


def decode(p: Float[Array, "k*(m*(d+1)+d)"], d: int, k: int) -> rkhs.Function:
    """Decode a candidate that carries its own lengthscales in the trailing k*d entries."""
    # the lengthscales are optimized in log scale
    log_lo, log_hi = np.log(rkhs.RHO_RANGE[0]), np.log(rkhs.RHO_RANGE[1])
    rho = jnp.exp(p[-k * d :].reshape(k, d) * (log_hi - log_lo) + log_lo)
    return rkhs.Function.from_array(rho, p[: -k * d].reshape(k, -1, d + 1))


def negative_log_ei(p: Float[Array, "k*(m*(d+1)+d)"], posterior, d: int, k: int) -> Scalar:
    """EI of the function parametrized by basis points plus its own lengthscales."""
    mu, cov = posterior.predict([decode(p, d, k)])
    return -acquisition.log_expected_improvement(
        mu=mu.squeeze(), sigma=cov.squeeze() ** 0.5, y_best=posterior.y_best
    )


@eqx.filter_jit
def screening_loss(ps, posterior, d: int, k: int) -> Float[Array, "n"]:
    return jax.vmap(negative_log_ei, in_axes=(0, None, None, None))(ps, posterior, d, k)


acquisition_loss = eqx.filter_jit(jax.value_and_grad(negative_log_ei))


def run(
    seed: int,
    target_fn: targets.TestFunction,
    surrogate_model: gp.FunctionalGaussianProcess,
    initial_acquisitions: int,
    minimum_m: int,
    maximum_m: int,
    acquisitions_each_m: int,
    acquisition_raw_samples: int,
    acquisition_max_restarts: int,
) -> dict:
    """Sequential EI loop over the RKHS parametrization, one basis point budget at a time."""
    rng = np.random.default_rng(seed=seed)
    d, k = target_fn.d, getattr(target_fn, "m", 1)
    timings: dict[str, float] = defaultdict(float)

    fs: list[rkhs.Function] = []
    ys: Float[Array, "n"] = jnp.zeros(0)

    def record(new_fs: list[rkhs.Function]) -> None:
        nonlocal fs, ys, surrogate_model
        timer = time.perf_counter()
        new_ys = jnp.array([target_fn(f) for f in new_fs])
        timings["target_evaluation"] += time.perf_counter() - timer

        fs, ys = fs + list(new_fs), jnp.concatenate([ys, new_ys])
        timer = time.perf_counter()
        # observations are append-only, so the previous distance block stays valid
        surrogate_model = surrogate_model.fit(fs, ys, cached_dists=surrogate_model.d2)
        timings["surrogate_fit"] += time.perf_counter() - timer

    # every output gets m basis points of d+1 entries, plus its own d lengthscales
    sampler = lambda m: sp.stats.qmc.LatinHypercube(d=k * (m * (d + 1) + d), rng=rng)

    timer = time.perf_counter()
    candidate_sampler = sampler(minimum_m)
    initial_fs = [
        decode(jnp.asarray(p), d, k)
        for p in candidate_sampler.random(n=initial_acquisitions)
    ]
    timings["acquisition"] += time.perf_counter() - timer
    record(initial_fs)

    for m in range(minimum_m, maximum_m + 1):
        candidate_sampler = sampler(m)
        for i in range(acquisitions_each_m):
            timer = time.perf_counter()
            ps = jnp.array(candidate_sampler.random(n=acquisition_raw_samples))
            posterior = surrogate_model.posterior
            p, _ = acquisition.optimize_lhs_candidates(
                acquisition_loss=acquisition_loss,
                loss_args=(posterior, d, k),
                candidates=ps,
                max_restarts=acquisition_max_restarts,
                screening_loss=lambda ps: screening_loss(ps, posterior, d, k),
            )
            timings["acquisition"] += time.perf_counter() - timer

            record([decode(p, d, k)])
            vprint(
                f"Iteration {i+1} (m={m}): "
                f"current = {ys[-1]:.8f}, best = {ys.min():.8f}\n"
            )

    return dict(
        observation_locations=fs,
        observation_values=ys,
        **{f"{stage}_time": t for stage, t in timings.items()},
    )


def main():
    global VERBOSE

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_fn", choices=list(targets.TARGET_FNS))
    parser.add_argument(
        "--profile", choices=["rbf", "matern52", "matern32", "matern12"]
    )
    parser.add_argument("--seed", type=int, required=True)
    # simulation parameters
    parser.add_argument("--initial_acquisitions", type=int, default=10)
    parser.add_argument("--minimum_m", type=int, default=1)
    parser.add_argument("--maximum_m", type=int, default=10)
    parser.add_argument("--acquisitions_each_m", type=int, default=10)
    parser.add_argument("--acquisition_raw_samples", type=int, default=1024)
    parser.add_argument("--acquisition_max_restarts", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose

    target_fn = targets.make_target(args.target_fn)
    profile = {
        "rbf": kernels.squared_exponential,
        "matern12": kernels.matern12,
        "matern32": kernels.matern32,
        "matern52": kernels.matern52,
    }[args.profile]
    # adaptive candidates need an ambient below their range, so that l1 + l2 - l0 > 0
    ambient_rho = jnp.full(target_fn.d, rkhs.RHO_RANGE[0])

    results = run(
        seed=args.seed,
        target_fn=target_fn,
        surrogate_model=gp.FunctionalGaussianProcess(profile=profile, ambient_rho=ambient_rho),
        initial_acquisitions=args.initial_acquisitions,
        minimum_m=args.minimum_m,
        maximum_m=args.maximum_m,
        acquisitions_each_m=args.acquisitions_each_m,
        acquisition_raw_samples=args.acquisition_raw_samples,
        acquisition_max_restarts=args.acquisition_max_restarts,
    )

    save_dir = f"{RESULTS_DIR}/{args.target_fn}/ours_adaptive/{args.profile}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}"
    pickle.dump(downcast(results), open(f"{save_path}.pkl", "wb"))
    vprint(f"Results saved to {save_path}.pkl")


if __name__ == "__main__":
    main()
