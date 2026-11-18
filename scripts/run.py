import argparse
import os
from functools import partial

import numpy as np

from jaxtyping import Array, Float, Scalar
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from bofus import gp, kernels, rkhs, acquisition
from targets import (
    TestFunction,
    Ridge,
    Projection,
    gymnasium,
)
from vlse import (
    GramacyLee,
    Ackley,
    Hartmann3,
    Rosenbrock,
    Michalewicz,
)


def expand_to(fs: rkhs.RBFMixture["n o m d"], ys: Float[Array, "n"], size: int):
    """Grow the observation axis to size, filling new slots with zeroed copies of the first."""

    def expand(x: Float[Array, "n ..."], fill: Array) -> Float[Array, "size ..."]:
        return jnp.concat([x, jnp.broadcast_to(fill, (size - len(x), *fill.shape))])

    first = jax.tree.map(lambda z: z[0], fs)
    fs = jax.tree.map(expand, fs, first._replace(a=jnp.zeros_like(first.a)))
    ys = expand(ys, jnp.asarray(jnp.nan))
    return fs, ys


def run(
    seed: int,
    target_fn: TestFunction,
    profile: kernels.Profile,
    m: int,
    initial_acquisitions: int,
    budget: int,
    batch_size: int = 1,
    l_range: tuple[float, float | None] = (1e-3, None),
    a_range: tuple[Scalar, Scalar] = (jnp.asarray(-3.0), jnp.asarray(3.0)),
    fixed_l: float | None = None,
    distance: str = "rkhs",
    save_path: str | None = None,
    checkpoint_every: int = 8,
    resume: tuple[rkhs.RBFMixture, Float[Array, "n"]] | None = None,
    verbose: bool = False,
) -> tuple[rkhs.RBFMixture["n o m d"], Float[Array, "n"]]:
    """Sequential EI loop over m atom mixtures until budget observations.
    Resume continues from a checkpointed (fs, ys) as if never interrupted.
    """
    key, key_init = jr.split(jr.key(seed))
    d, k = target_fn.d, target_fn.k
    # no cap defaults to d: the coverage lengthscale of the prior grows linearly with d
    l_min, l_max = (fixed_l, fixed_l) if fixed_l is not None else l_range
    l_range = (jnp.asarray(l_min), jnp.asarray(d if l_max is None else l_max, dtype=float))

    # closed loop control marks inputs folded from (sin, cos) pairs as wrapping
    periodic = getattr(target_fn, "periodic", None)
    # rkhs distance at the smallest lengthscale, or L2 on the box
    l0 = jnp.broadcast_to(l_range[0], (d,)) if distance == "rkhs" else None

    if resume is None:
        # initial design from the mixture prior
        shape = (initial_acquisitions, k, m, d)
        fs = rkhs.RBFMixture.sample(key_init, shape, l_range, a_range)
        ys = jnp.asarray(
            [
                target_fn(jax.tree.map(lambda z: z[i], fs), periodic=periodic)
                for i in range(initial_acquisitions)
            ]
        )
    else:
        # replay the key splits of the acquisitions already made (one per batch)
        fs, ys = resume
        for _ in range(-(-(len(ys) - initial_acquisitions) // batch_size)):
            key, _ = jr.split(key)

    i = len(ys)
    while i < budget:
        # grow the buffers in powers of two, so each size compiles once
        if i == len(ys):
            fs, ys = expand_to(fs, ys, min(budget, 1 << i.bit_length()))

        key, key_step = jr.split(key)
        key_fit, key_acq = jr.split(key_step)
        surrogate = gp.GaussianProcess.fit(
            key_fit, fs, ys, profile=profile, l0=l0, periodic=periodic
        )
        f_batch = acquisition.optimize_expected_improvement(
            key_acq, surrogate, l_range, a_range, batch_size=batch_size
        )

        # Evaluate and store each candidate, truncating the batch at the buffer end
        for j in range(min(batch_size, len(ys) - i)):
            f = jax.tree.map(lambda z: z[j], f_batch)
            fs = jax.tree.map(lambda z, w: z.at[i].set(w), fs, f)
            ys = ys.at[i].set(target_fn(f, periodic=periodic))

            if verbose:
                print(f"Iteration {i + 1}: current = {ys[i]:.6g}, best = {jnp.nanmin(ys):.6g}")
            i += 1

        # checkpoint every few acquisitions so long runs are inspectable mid-flight
        if save_path is not None and (i % checkpoint_every == 0 or i == budget):
            optimum = getattr(target_fn, "optimum", None)
            optimum = np.nan if optimum is None else optimum
            np.savez(
                save_path,
                l=fs.l[:i],
                x=fs.x[:i],
                a=fs.a[:i],
                y=ys[:i],
                optimum=optimum,
            )

    return jax.tree.map(lambda z: z[:i], fs), ys[:i]


def suite_of(target: str) -> str:
    """Results folder grouping a target with its benchmark suite."""
    if target.endswith("_ridge"):
        return "ridges"
    if target.endswith("_projection"):
        return "projections"
    return "openloop" if target.endswith("-open-loop") else "closedloop"


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
    **gymnasium.ENVS,
    **{
        f"{env_id}-open-loop": partial(cls, open_loop=True)
        for env_id, cls in gymnasium.ENVS.items()
    },
)


if __name__ == "__main__":
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
    parser.add_argument("--distance", choices=["rkhs", "l2"], default="rkhs")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--initial_acquisitions", type=int, default=8)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--fixed_l",
        type=float,
        default=None,
        help="share this squared lengthscale across atoms (non-adaptive ablation)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results_dir", default="results/neurips")
    args = parser.parse_args()

    target_fn = target_fns[args.target_fn]()
    profile = profiles_options[args.profile]

    method = "adaptive" if args.fixed_l is None else f"fixed_l{args.fixed_l:g}"
    method = f"ours_{args.distance}_{method}"
    save_dir = f"{args.results_dir}/{suite_of(args.target_fn)}/{args.target_fn}/{method}/{args.profile}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}_m_{args.m}.npz"

    resume = None
    if os.path.exists(save_path):
        checkpoint = np.load(save_path)
        if len(checkpoint["y"]) >= args.budget:
            print(f"already complete: {save_path}")
            raise SystemExit(0)
        if len(checkpoint["y"]) >= args.initial_acquisitions:
            fs = rkhs.RBFMixture(*(jnp.asarray(checkpoint[z]) for z in ("l", "x", "a")))
            resume = fs, jnp.asarray(checkpoint["y"])
            print(f"resuming from {len(checkpoint['y'])} acquisitions: {save_path}")

    run(
        seed=args.seed,
        target_fn=target_fn,
        profile=profile,
        m=args.m,
        initial_acquisitions=args.initial_acquisitions,
        budget=args.budget,
        batch_size=args.batch_size,
        fixed_l=args.fixed_l,
        distance=args.distance,
        save_path=save_path,
        resume=resume,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Results saved to {save_dir}")
