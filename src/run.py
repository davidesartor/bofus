from collections import defaultdict
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp

import argparse
import time
import os
import pickle

from bofus import gp, kernels, rkhs, acquisition
import targets
import vlse

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/neurips")

VERBOSE = False

RHO_RANGE = (0.05, 0.4)  # candidate lengthscales, same range the sweeps scan

TARGET_FNS = {
    "gramacylee": lambda: targets.Ridge(vlse.GramacyLee(normalized=True)),
    "ackley": lambda: targets.Ridge(vlse.Ackley(d=2, normalized=True)),
    "hartmann": lambda: targets.Ridge(vlse.Hartmann3(normalized=True)),
    "rosenbrock": lambda: targets.Ridge(vlse.Rosenbrock(d=4, normalized=True)),
    "michalewicz": lambda: targets.Ridge(vlse.Michalewicz(d=5, normalized=True)),
    "pendulum": targets.Pendulum,
    "pinwheel": targets.PinWheel,
    "brachistochrone": targets.Brachistochrone,
    "hopper": targets.HoppingRobot,
    "mnist": targets.MNIST,
}


def run(
    seed: int,
    target_fn: targets.TestFunction,
    profile: kernels.Profile,
    m: int,
    initial_acquisitions: int,
    n_acquisitions: int,
) -> dict:
    """Sequential EI loop over the RKHS parametrization with a fixed basis size."""
    key = jax.random.key(seed)
    d, k = target_fn.d, getattr(target_fn, "m", 1)
    timings: dict[str, float] = defaultdict(float)

    # candidate ranges: squared lengthscales, basis points in the unit box, values in [-1,1]
    l_range = (jnp.asarray(RHO_RANGE[0] ** 2), jnp.asarray(RHO_RANGE[1] ** 2))
    x_range = (jnp.asarray(0.0), jnp.asarray(1.0))
    y_range = (jnp.asarray(-1.0), jnp.asarray(1.0))

    # padded buffers: nan observations mark unused capacity
    fs = rkhs.RBFMixture(
        l=jnp.ones((0, k, m, d)), x=jnp.zeros((0, k, m, d)), a=jnp.zeros((0, k, m))
    )
    ys: Float[Array, "n"] = jnp.zeros(0)
    n = 0

    def record(new_fs: rkhs.RBFMixture) -> gp.GaussianProcess:
        """Evaluate a batch of candidates, write them into the buffers, refit the surrogate."""
        nonlocal fs, ys, n
        timer = time.perf_counter()
        rows = [jax.tree.map(lambda z: z[i], new_fs) for i in range(new_fs.a.shape[0])]
        new_ys = jnp.array([target_fn(f) for f in rows])
        timings["target_evaluation"] += time.perf_counter() - timer

        timer = time.perf_counter()

        # grow to the next power of two when out of capacity, so the fit rarely retraces
        n_new = len(new_ys)
        if n + n_new > len(ys):
            N = 1 << (n + n_new - 1).bit_length()
            pad = lambda z, fill: jnp.concat([z, jnp.full((N - len(ys), *z.shape[1:]), fill)])
            fs = rkhs.RBFMixture(l=pad(fs.l, 1.0), x=pad(fs.x, 0.0), a=pad(fs.a, 0.0))
            ys = pad(ys, jnp.nan)

        fs = jax.tree.map(lambda z, w: z.at[n : n + n_new].set(w), fs, new_fs)
        ys = ys.at[n : n + n_new].set(new_ys)
        n += n_new

        surrogate = gp.GaussianProcess.fit(fs, ys, profile=profile)
        timings["surrogate_fit"] += time.perf_counter() - timer
        return surrogate

    timer = time.perf_counter()
    key, init_key = jax.random.split(key)
    initial = rkhs.RBFMixture.from_lhs(
        init_key, (initial_acquisitions, k, m, d), l_range, x_range, y_range
    )
    timings["acquisition"] += time.perf_counter() - timer
    surrogate_model = record(initial)

    for i in range(n_acquisitions):
        timer = time.perf_counter()
        key, acq_key = jax.random.split(key)
        f = acquisition.optimize_expected_improvement(
            acq_key, surrogate_model, l_range, x_range, y_range
        )
        timings["acquisition"] += time.perf_counter() - timer

        surrogate_model = record(jax.tree.map(lambda z: z[None], f))
        if VERBOSE:
            print(f"Iteration {i+1}: current = {ys[n-1]:.8f}, best = {jnp.nanmin(ys):.8f}\n")

    return dict(
        observation_locations=jax.tree.map(lambda z: z[:n], fs),
        observation_values=ys[:n],
        **{f"{stage}_time": t for stage, t in timings.items()},
    )


def main():
    global VERBOSE

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_fn", choices=list(TARGET_FNS), required=True)
    parser.add_argument(
        "--profile", choices=["rbf", "matern52", "matern32", "matern12"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    # simulation parameters
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--initial_acquisitions", type=int, default=10)
    parser.add_argument("--n_acquisitions", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose

    target_fn = TARGET_FNS[args.target_fn]()
    profile = {
        "rbf": kernels.rbf,
        "matern12": kernels.matern12,
        "matern32": kernels.matern32,
        "matern52": kernels.matern52,
    }[args.profile]

    results = run(
        seed=args.seed,
        target_fn=target_fn,
        profile=profile,
        m=args.m,
        initial_acquisitions=args.initial_acquisitions,
        n_acquisitions=args.n_acquisitions,
    )

    save_dir = f"{RESULTS_DIR}/{args.target_fn}/ours_adaptive/{args.profile}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}_m_{args.m}"
    pickle.dump(results, open(f"{save_path}.pkl", "wb"))
    if VERBOSE:
        print(f"Results saved to {save_path}.pkl")


if __name__ == "__main__":
    main()
