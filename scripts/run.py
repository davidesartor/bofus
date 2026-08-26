import argparse
import os
from functools import partial

import numpy as np

from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
import jax.random as jr

from bofus import gp, kernels, rkhs, acquisition
from targets import (
    TestFunction,
    Ridge,
    Projection,
    Pendulum,
    PinWheel,
    Brachistochrone,
    HoppingRobot,
    MNIST,
)
from vlse import (
    GramacyLee,
    Ackley,
    Hartmann3,
    Rosenbrock,
    Michalewicz,
)


def expand_to(fs: rkhs.RBFMixture, ys: Float[Array, "o"], size: int):
    """Grow the observation axis to size, filling new slots inert."""

    def expand(x: Float[Array, "o ..."], fill: float) -> Float[Array, "n ..."]:
        o, *rest = x.shape
        return jnp.full((size, *rest), fill).at[:o].set(x)

    fills = rkhs.RBFMixture(l=1.0, x=0.0, a=0.0)
    fs = jax.tree.map(expand, fs, fills)
    ys = expand(ys, jnp.nan)
    return fs, ys


def run(
    seed: int,
    target_fn: TestFunction,
    profile: kernels.Profile,
    m: int,
    initial_acquisitions: int,
    batch_size: int = 1,
    l_range: tuple[Scalar, Scalar] = (jnp.asarray(0.01), jnp.asarray(1.0)),
    x_range: tuple[Scalar, Scalar] = (jnp.asarray(0.0), jnp.asarray(1.0)),
    a_range: tuple[Scalar, Scalar] = (jnp.asarray(-1.0), jnp.asarray(1.0)),
    save_path: str | None = None,
    verbose: bool = False,
) -> dict:
    """Sequential EI loop doubling the basis size, each stage doubling the observations."""
    key = jr.key(seed)
    d, k = target_fn.d, target_fn.k
    schedule = [1 << s for s in range((m - 1).bit_length())] + [m]

    # initialize the observation buffers with the evaluated latin hypercube sample
    key, key_init = jr.split(key)
    fs = rkhs.RBFMixture.from_lhs(
        key_init, (initial_acquisitions, k, schedule[0], d), l_range, x_range, a_range
    )
    ys = [
        target_fn(jax.tree.map(lambda z: z[i], fs)) for i in range(initial_acquisitions)
    ]
    ys = jnp.asarray(ys)

    i = initial_acquisitions
    for m_stage in schedule:
        # grow both buffers at the stage boundary, so each stage compiles once
        stage_end = 2 * i
        fs = fs if m_stage == fs.l.shape[-2] else fs.split()
        fs, ys = expand_to(fs, ys, stage_end)
        while i < stage_end:
            # Fit the GP surrogate model to the current observations
            surrogate = gp.GaussianProcess.fit(fs, ys, profile=profile)

            # Optimize the acquisition function to find the next batch to evaluate
            key, key_acq = jr.split(key)
            f_batch = acquisition.optimize_expected_improvement(
                key_acq,
                surrogate,
                l_range,
                x_range,
                a_range,
                batch_size=batch_size,
            )

            # Evaluate and store each candidate, truncating the batch at the budget
            for j in range(min(batch_size, stage_end - i)):
                f = jax.tree.map(lambda z: z[j], f_batch)
                y = target_fn(f)
                fs = jax.tree.map(lambda z, w: z.at[i].set(w), fs, f)
                ys = ys.at[i].set(y)

                if verbose:
                    print(
                        f"Iteration {i + 1} (m={m_stage}): "
                        f"current = {ys[i]:.8f}, best = {jnp.nanmin(ys):.8f}\n"
                    )
                i += 1

        # checkpoint the completed stage so long runs are inspectable mid-flight
        if save_path is not None:
            np.savez(save_path, l=fs.l[:i], x=fs.x[:i], a=fs.a[:i], y=ys[:i])

    total_acquisitions = i

    return dict(
        observation_locations=jax.tree.map(lambda z: z[:total_acquisitions], fs),
        observation_values=ys[:total_acquisitions],
    )


if __name__ == "__main__":
    target_fns = dict(
        gramacylee_ridge=partial(Ridge, GramacyLee(normalized=True)),
        ackley_ridge=partial(Ridge, Ackley(d=2, normalized=True)),
        hartmann_ridge=partial(Ridge, Hartmann3(normalized=True)),
        rosenbrock_ridge=partial(Ridge, Rosenbrock(d=4, normalized=True)),
        michalewicz_ridge=partial(Ridge, Michalewicz(d=5, normalized=True)),
        gramacylee_projection=partial(Projection, GramacyLee(normalized=True)),
        ackley_projection=partial(Projection, Ackley(d=2, normalized=True)),
        hartmann_projection=partial(Projection, Hartmann3(normalized=True)),
        rosenbrock_projection=partial(Projection, Rosenbrock(d=4, normalized=True)),
        michalewicz_projection=partial(Projection, Michalewicz(d=5, normalized=True)),
        pendulum=Pendulum,
        pinwheel=PinWheel,
        brachistochrone=Brachistochrone,
        hopper=HoppingRobot,
        mnist=MNIST,
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
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--initial_acquisitions", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results_dir", default="results/neurips")
    args = parser.parse_args()

    target_fn = target_fns[args.target_fn]()
    profile = profiles_options[args.profile]

    save_dir = f"{args.results_dir}/{args.target_fn}/ours_adaptive/{args.profile}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}_m_{args.m}.npz"

    results = run(
        seed=args.seed,
        target_fn=target_fn,
        profile=profile,
        m=args.m,
        initial_acquisitions=args.initial_acquisitions,
        batch_size=args.batch_size,
        save_path=save_path,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Results saved to {save_path}")
