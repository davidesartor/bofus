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
    MujocoTask,
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
    fixed_l: float | None = None,
    distance: gp.Distance = "rkhs",
    save_path: str | None = None,
    verbose: bool = False,
) -> dict:
    """Sequential EI loop doubling the basis size, each stage doubling the observations.

    fixed_l shares one squared lengthscale across all atoms instead of adapting it.
    """
    key = jr.key(seed)
    if fixed_l is not None:
        l_range = (jnp.asarray(fixed_l), jnp.asarray(fixed_l))
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
    surrogate = None
    for m_stage in schedule:
        # grow both buffers at the stage boundary, so each stage compiles once
        stage_end = 2 * i
        fs = fs if m_stage == fs.l.shape[-2] else fs.split()
        fs, ys = expand_to(fs, ys, stage_end)
        while i < stage_end:
            # Fit the GP surrogate model, warm started from the previous fit
            key, key_fit, key_acq = jr.split(key, 3)
            init = None if surrogate is None else (surrogate.l0, surrogate.rho, surrogate.g)
            surrogate = gp.GaussianProcess.fit(
                fs, ys, profile=profile, init=init, key=key_fit, distance=distance
            )

            # Optimize the acquisition function to find the next batch to evaluate
            f_batch = acquisition.optimize_expected_improvement(
                key_acq,
                surrogate,
                l_range,
                x_range,
                a_range,
                batch_size=batch_size,
                fixed_l=fixed_l,
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
            optimum = getattr(target_fn, "optimum", None)
            optimum = np.nan if optimum is None else optimum
            np.savez(
                save_path, l=fs.l[:i], x=fs.x[:i], a=fs.a[:i], y=ys[:i], optimum=optimum
            )

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
        **{
            f"{name}_{mode}": partial(MujocoTask, env_id, mode)
            for name, env_id, modes in [
                ("inverted_pendulum", "InvertedPendulum-v5", ["closed"]),
                ("inverted_double_pendulum", "InvertedDoublePendulum-v5", ["closed"]),
                ("reacher", "Reacher-v5", ["closed"]),
                ("swimmer", "Swimmer-v5", ["open", "closed"]),
                ("mj_hopper", "Hopper-v5", ["open", "closed"]),
                ("walker2d", "Walker2d-v5", ["open", "closed"]),
                ("half_cheetah", "HalfCheetah-v5", ["open", "closed"]),
            ]
            for mode in modes
        },
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
    parser.add_argument(
        "--fixed_l",
        type=float,
        default=None,
        help="share this squared lengthscale across atoms (non-adaptive ablation)",
    )
    parser.add_argument(
        "--distance",
        choices=["rkhs", "l2"],
        default="rkhs",
        help="distance between functions in the GP kernel",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results_dir", default="results/neurips")
    args = parser.parse_args()

    target_fn = target_fns[args.target_fn]()
    profile = profiles_options[args.profile]

    method = "ours_adaptive" if args.fixed_l is None else f"ours_fixed_l{args.fixed_l:g}"
    method += "" if args.distance == "rkhs" else f"_{args.distance}"
    save_dir = f"{args.results_dir}/{args.target_fn}/{method}/{args.profile}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}_m_{args.m}.npz"

    results = run(
        seed=args.seed,
        target_fn=target_fn,
        profile=profile,
        m=args.m,
        initial_acquisitions=args.initial_acquisitions,
        batch_size=args.batch_size,
        fixed_l=args.fixed_l,
        distance=args.distance,
        save_path=save_path,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Results saved to {save_path}")
