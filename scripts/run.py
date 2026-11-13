import argparse
import os
from functools import partial

import numpy as np
from einops import rearrange
from scipy.stats import qmc

from jaxtyping import Array, Float, Scalar
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from bofus import gp, kernels, rkhs, acquisition, utils
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
    budget: int | None = None,
    batch_size: int = 1,
    l_range: tuple[Scalar, Scalar] = (jnp.asarray(0.04), jnp.asarray(4.0)),
    fixed_l: float | None = None,
    distance: str = "rkhs",
    save_path: str | None = None,
    checkpoint_every: int = 8,
    resume: tuple[rkhs.RBFMixture, Float[Array, "n"]] | None = None,
    verbose: bool = False,
) -> dict:
    """Sequential EI loop doubling the basis size, each stage doubling the observations.

    With a budget, the last stage (m atoms) runs on until budget observations.
    Resume continues from checkpointed (fs, ys) as if never interrupted.
    """
    key = jr.key(seed)
    if fixed_l is not None:
        l_range = (jnp.asarray(fixed_l), jnp.asarray(fixed_l))
    d, k = target_fn.d, target_fn.k

    # closed loop control marks inputs folded from (sin, cos) pairs as wrapping
    periodic = getattr(target_fn, "periodic", None)
    # rkhs distance at the smallest lengthscale, or L2 on the box
    l0 = jnp.broadcast_to(l_range[0], (d,)) if distance == "rkhs" else None

    schedule = [1 << s for s in range((m - 1).bit_length())] + [m]

    # sobol design over log lengthscales, centers and amplitudes of the first stage
    sobol = qmc.Sobol(k * schedule[0] * (2 * d + 1), seed=seed)
    p = jnp.asarray(sobol.random(initial_acquisitions))
    p = rearrange(p, "n (o m p) -> n o m p", o=k, m=schedule[0])
    log_l, x, a = jnp.split(p, [d, 2 * d], axis=-1)
    fs = rkhs.RBFMixture(
        jnp.exp(utils.rescale(log_l, jnp.log(l_range[0]), jnp.log(l_range[1]))),
        utils.rescale(x, -rkhs.X_MAX, rkhs.X_MAX),
        utils.rescale(a.squeeze(-1), -acquisition.A_MAX, acquisition.A_MAX),
    )

    if resume is None:
        # initialize the observation buffers with the evaluated design
        ys = [
            target_fn(jax.tree.map(lambda z: z[i], fs), periodic=periodic)
            for i in range(initial_acquisitions)
        ]
        ys = jnp.asarray(ys)
    else:
        # replay the key splits of the acquisitions already made (one per batch)
        fs, ys = resume
        for _ in range(-(-(len(ys) - initial_acquisitions) // batch_size)):
            key, _, _ = jr.split(key, 3)

    i, stage_start = len(ys), initial_acquisitions
    for m_stage in schedule:
        # grow both buffers at the stage boundary, so each stage compiles once
        stage_end = budget if budget is not None and m_stage == m else 2 * stage_start
        stage_start = stage_end
        if i >= stage_end:
            continue
        if m_stage != fs.l.shape[-2]:
            fs = fs.pad_to(m_stage)
        fs, ys = expand_to(fs, ys, stage_end)
        while i < stage_end:
            # Fit the GP surrogate model
            key, key_fit, key_acq = jr.split(key, 3)
            surrogate = gp.GaussianProcess.fit(
                key_fit, fs, ys, profile=profile, l0=l0, periodic=periodic
            )

            # Optimize the acquisition function to find the next batch to evaluate
            f_batch = acquisition.optimize_expected_improvement(
                key_acq,
                surrogate,
                l_range,
                batch_size=batch_size,
            )

            # Evaluate and store each candidate, truncating the batch at the budget
            for j in range(min(batch_size, stage_end - i)):
                f = jax.tree.map(lambda z: z[j], f_batch)
                y = target_fn(f, periodic=periodic)
                fs = jax.tree.map(lambda z, w: z.at[i].set(w), fs, f)
                ys = ys.at[i].set(y)

                if verbose:
                    print(
                        f"Iteration {i + 1} (m={m_stage}): "
                        f"current = {ys[i]:.8f}, best = {jnp.nanmin(ys):.8f}\n"
                    )
                i += 1

            # checkpoint every few acquisitions so long runs are inspectable mid-flight
            if save_path is not None and (i % checkpoint_every == 0 or i == stage_end):
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

    total_acquisitions = i

    return dict(
        observation_locations=jax.tree.map(lambda z: z[:total_acquisitions], fs),
        observation_values=ys[:total_acquisitions],
    )


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
    parser.add_argument("--budget", type=int, default=None)
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
    save_dir = f"{args.results_dir}/{args.target_fn}/{method}/{args.profile}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}_m_{args.m}.npz"

    # pick up from the last checkpoint, if any
    resume = None
    if os.path.exists(save_path):
        checkpoint = np.load(save_path)
        if args.budget is not None and len(checkpoint["y"]) >= args.budget:
            raise SystemExit(f"already complete: {save_path}")
        fs = rkhs.RBFMixture(
            *(jnp.asarray(checkpoint[name]) for name in ("l", "x", "a"))
        )
        resume = fs, jnp.asarray(checkpoint["y"])
        print(f"resuming from {len(resume[1])} acquisitions: {save_path}")

    results = run(
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
        print(f"Results saved to {save_path}")
