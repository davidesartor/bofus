import argparse
import os
from functools import partial

import numpy as np

from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from bofus import gp, kernels, rkhs, acquisition
from targets import (
    TestFunction,
    Ridge,
    Projection,
    Pendulum,
    PinWheel,
    Brachistochrone,
    HoppingRobot,
    Whip,
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
    budget: int | None = None,
    batch_size: int = 1,
    l_range: tuple[Scalar, Scalar] = (jnp.asarray(0.01), jnp.asarray(1.0)),
    x_range: tuple[Scalar, Scalar] = (jnp.asarray(0.0), jnp.asarray(1.0)),
    a_range: tuple[Scalar, Scalar] = (jnp.asarray(-1.0), jnp.asarray(1.0)),
    save_path: str | None = None,
    verbose: bool = False,
    adaptive: bool = True,
    log_objective: bool = False,
    log_eps: float = 1e-2,
) -> dict:
    """Sequential EI loop doubling the basis size, each stage doubling the observations.

    The last stage runs to budget when given, else every stage doubles. With
    log_objective the GP is fit on log(y + log_eps), results keep the raw values.
    """
    key = jr.key(seed)
    d, k = target_fn.d, target_fn.k
    schedule = [1 << s for s in range((m - 1).bit_length())] + [m] if adaptive else [m]

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
        last_stage = m_stage == schedule[-1]
        stage_end = 2 * i if budget is None else budget if last_stage else min(2 * i, budget)
        fs = fs if m_stage == fs.l.shape[-2] else fs.split()
        fs, ys = expand_to(fs, ys, stage_end)
        while i < stage_end:
            # Fit the GP surrogate model, warm started from the previous fit
            key, key_fit, key_acq = jr.split(key, 3)
            init = None if surrogate is None else (surrogate.l0, surrogate.rho, surrogate.g)
            ys_fit = jnp.log(ys + log_eps) if log_objective else ys
            surrogate = gp.GaussianProcess.fit(
                fs, ys_fit, profile=profile, init=init, key=key_fit
            )

            # Optimize the acquisition function to find the next batch to evaluate
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

            # checkpoint every few acquisitions so long runs are inspectable mid-flight
            if save_path is not None and i % 8 == 0:
                np.savez(save_path, l=fs.l[:i], x=fs.x[:i], a=fs.a[:i], y=ys[:i])

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
        whip=Whip,
        **{f"whip3d_{i}": partial(Whip, target=i) for i in range(1, 7)},
        whip2d=partial(Whip, planar=True),
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
    parser.add_argument(
        "--budget", type=int, default=None, help="Total acquisitions, else every stage doubles."
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fixed_m", action="store_true", help="Keep the basis size fixed at m.")
    parser.add_argument(
        "--log_objective", action="store_true", help="Fit the GP on log(y + 1e-2)."
    )
    parser.add_argument("--results_dir", default="results/icra")
    args = parser.parse_args()

    target_fn = target_fns[args.target_fn]()
    profile = profiles_options[args.profile]

    method = "rbf_ei_adaptive" if not args.fixed_m else "rbf_ei"
    save_dir = f"{args.results_dir}/{args.target_fn}/{method}/{args.profile}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}_m_{args.m}.npz"

    results = run(
        seed=args.seed,
        target_fn=target_fn,
        profile=profile,
        m=args.m,
        initial_acquisitions=args.initial_acquisitions,
        budget=args.budget,
        batch_size=args.batch_size,
        save_path=save_path,
        verbose=args.verbose,
        adaptive=not args.fixed_m,
        log_objective=args.log_objective,
    )

    if args.verbose:
        print(f"Results saved to {save_path}")
