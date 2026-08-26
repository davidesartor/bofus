import argparse
import os
from functools import partial

import numpy as np

from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
import jax.random as jr

from bofus import gp, kernels, rkhs, acquisition
import targets
import vlse


def maybe_expand(fs: rkhs.RBFMixture, ys: Float[Array, "o"]):
    """Expand the buffers if they are full."""

    def expand(x: Float[Array, "o ..."], fill: float) -> Float[Array, "n ..."]:
        o, *rest = x.shape
        n = 1 << o.bit_length()
        return jnp.full((n, *rest), fill).at[:o].set(x)

    if not jnp.isnan(ys).any():
        fills = rkhs.RBFMixture(l=1.0, x=0.0, a=0.0)
        fs = jax.tree.map(expand, fs, fills)
        ys = expand(ys, jnp.nan)
    return fs, ys


def run(
    seed: int,
    target_fn: targets.TestFunction,
    profile: kernels.Profile,
    m: int,
    initial_acquisitions: int,
    total_acquisitions: int,
    l_range: tuple[Scalar, Scalar] = (jnp.asarray(0.01), jnp.asarray(1.0)),
    x_range: tuple[Scalar, Scalar] = (jnp.asarray(0.0), jnp.asarray(1.0)),
    y_range: tuple[Scalar, Scalar] = (jnp.asarray(-1.0), jnp.asarray(1.0)),
    verbose: bool = False,
) -> dict:
    """Sequential EI loop over the RKHS parametrization with a fixed basis size."""
    key = jr.key(seed)
    d, k = target_fn.d, target_fn.k

    # initialize the observation buffers with the evaluated latin hypercube sample
    key, key_init = jr.split(key)
    fs = rkhs.RBFMixture.from_lhs(
        key_init, (initial_acquisitions, k, m, d), l_range, x_range, y_range
    )
    ys = [
        target_fn(jax.tree.map(lambda z: z[i], fs)) for i in range(initial_acquisitions)
    ]
    ys = jnp.asarray(ys)

    for i in range(initial_acquisitions, total_acquisitions):
        # Fit the GP surrogate model to the current observations
        surrogate = gp.GaussianProcess.fit(fs, ys, profile=profile)

        # Optimize the acquisition function to find the next point to evaluate
        key, key_acq = jr.split(key)
        f = acquisition.optimize_expected_improvement(
            key_acq, surrogate, l_range, x_range, y_range
        )

        # Evaluate the target function at the new point
        y = target_fn(f)

        # Add the new observation to the buffers, expanding if necessary
        fs, ys = maybe_expand(fs, ys)
        fs = jax.tree.map(lambda z, w: z.at[i].set(w), fs, f)
        ys = ys.at[i].set(y)

        if verbose:
            print(
                f"Iteration {i + 1}: "
                f"current = {ys[i]:.8f}, best = {jnp.nanmin(ys):.8f}\n"
            )

    return dict(
        observation_locations=jax.tree.map(lambda z: z[:total_acquisitions], fs),
        observation_values=ys[:total_acquisitions],
    )


if __name__ == "__main__":
    vlse_targets = dict(
        gramacylee=vlse.GramacyLee(normalized=True),
        ackley=vlse.Ackley(d=2, normalized=True),
        hartmann=vlse.Hartmann3(normalized=True),
        rosenbrock=vlse.Rosenbrock(d=4, normalized=True),
        michalewicz=vlse.Michalewicz(d=5, normalized=True),
    )

    target_fns = dict(
        **{
            f"{name}_ridge": partial(targets.Ridge, function)
            for name, function in vlse_targets.items()
        },
        **{
            f"{name}_projection": partial(targets.Projection, function)
            for name, function in vlse_targets.items()
        },
        pendulum=targets.Pendulum,
        pinwheel=targets.PinWheel,
        brachistochrone=targets.Brachistochrone,
        hopper=targets.HoppingRobot,
        mnist=targets.MNIST,
    )

    profiles_options = dict(
        rbf=kernels.rbf,
        matern12=kernels.matern12,
        matern32=kernels.matern32,
        matern52=kernels.matern52,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_fn", choices=list(target_fns.keys()), required=True)
    parser.add_argument(
        "--profile", choices=list(profiles_options.keys()), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--initial_acquisitions", type=int, default=10)
    parser.add_argument("--n_acquisitions", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results_dir", default="results/neurips")
    args = parser.parse_args()

    target_fn = target_fns[args.target_fn]()
    profile = profiles_options[args.profile]

    results = run(
        seed=args.seed,
        target_fn=target_fn,
        profile=profile,
        m=args.m,
        initial_acquisitions=args.initial_acquisitions,
        total_acquisitions=args.initial_acquisitions + args.n_acquisitions,
        verbose=args.verbose,
    )

    save_dir = f"{args.results_dir}/{args.target_fn}/ours_adaptive/{args.profile}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}_m_{args.m}.npz"
    fs, ys = results["observation_locations"], results["observation_values"]
    np.savez(save_path, l=fs.l, x=fs.x, a=fs.a, y=ys)
    if args.verbose:
        print(f"Results saved to {save_path}")
