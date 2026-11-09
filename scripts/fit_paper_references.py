"""How sparse are the paper's min-jerk solutions: fit m-bump RBF mixtures to each reference and score them."""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy.optimize import least_squares

from bofus.rkhs import RBFMixture
from targets.whip import Whip, node_positions


def bumps(p: Float[Array, "3m"], t: Float[Array, "t"], m: int) -> Float[Array, "t"]:
    """Sum of m Gaussian bumps with squared widths l, centers x and amplitudes a, as RBFMixture does."""
    l, x, a = p[:m], p[m : 2 * m], p[2 * m :]
    return (a[None] * np.exp(-0.5 * (t[:, None] - x[None]) ** 2 / l[None])).sum(1)


def fit_joint(
    t: Float[Array, "t"], f: Float[Array, "t"], m: int, l_floor: float, starts: int, rng
) -> Float[Array, "3m"]:
    """Least-squares fit of m bumps within the BO search box, best of several random starts."""
    lo = np.r_[np.full(m, l_floor), np.zeros(m), np.full(m, -1.0)]
    hi = np.r_[np.ones(m), np.ones(m), np.ones(m)]
    best = None
    for _ in range(starts):
        p0 = np.r_[
            np.exp(rng.uniform(np.log(l_floor), 0.0, m)),
            rng.uniform(0.0, 1.0, m),
            rng.uniform(-1.0, 1.0, m),
        ]
        r = least_squares(lambda p: bumps(p, t, m) - f, p0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def hit_time(whip: Whip, q0, dq0) -> float:
    """When the tip nodes come closest to the target along the paper's reference."""
    _, qs, _ = whip.rollout(q0, dq0)
    nodes = jax.vmap(node_positions, in_axes=(0, None))(qs, whip.chain)
    dist = jnp.linalg.norm(nodes[:, -whip.n_tip_nodes :] - whip.target, axis=-1).min(1)
    return int(jnp.argmin(dist)) * whip.dt


def fit_reference(
    name: str, whip: Whip, index: int, ms: list[int], l_floors: list[float], starts: int, seed: int
) -> tuple[list[dict], dict]:
    """Fit every (l_floor, m) to the paper's reference on the window up to the hit, score both."""
    rng = np.random.default_rng(seed)
    ts, q0, dq0 = whip.paper_reference(index)
    t_hit = hit_time(whip, q0, dq0)
    cost_paper = float(whip.evaluate_reference(q0, dq0))

    # the reference in the BO's normalized coordinates, subsampled on the window that matters
    fstar = np.asarray((q0 - whip.reference_center) / whip.reference_amplitude)
    t = np.asarray(ts) / whip.horizon
    pre = t * whip.horizon <= t_hit
    sub = np.arange(0, len(t), 25)
    print(f"{name}: paper cost={cost_paper:.3f}, hit at {t_hit:.2f}s of {whip.horizon}s", flush=True)

    rows, params = [], {}
    for l_floor in l_floors:
        for m in ms:
            P = np.stack(
                [fit_joint(t[sub], fstar[sub, j], m, l_floor, starts, rng) for j in range(whip.k)]
            )
            mix = RBFMixture(
                l=jnp.asarray(P[:, :m, None]), x=jnp.asarray(P[:, m : 2 * m, None]), a=jnp.asarray(P[:, 2 * m :])
            )
            fit = np.stack([bumps(P[j], t, m) for j in range(whip.k)], 1)
            row = dict(
                reference=name,
                l_floor=l_floor,
                m=m,
                rmse_prehit=float(np.sqrt(((fit - fstar)[pre] ** 2).mean())),
                rmse_full=float(np.sqrt(((fit - fstar) ** 2).mean())),
                cost=float(whip(mix)),
                cost_paper=cost_paper,
                t_hit=t_hit,
                min_l=float(P[:, :m].min()),
            )
            rows.append(row)
            params[f"{name}/l_{l_floor}/m_{m}"] = P
            print(
                f"   l_floor={l_floor:<7} m={m}: rmse pre-hit={row['rmse_prehit']:.4f} "
                f"full={row['rmse_full']:.4f}, cost={row['cost']:.3f}, min l={row['min_l']:.5f}",
                flush=True,
            )
    return rows, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--l_floors", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--references", nargs="+", default=[f"whip{i}" for i in range(1, 7)] + ["whip2d"])
    parser.add_argument("--results_dir", default="results/paper_sparsity")
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    rows, params = [], {}
    for name in args.references:
        if name == "whip2d":
            whip, index = Whip(planar=True), 1
        else:
            index = int(name.removeprefix("whip"))
            whip = Whip(target=index)
        r, p = fit_reference(name, whip, index, args.ms, args.l_floors, args.starts, args.seed)
        rows += r
        params |= p
        with open(f"{args.results_dir}/fits.json", "w") as fh:
            json.dump(rows, fh, indent=1)
        np.savez(f"{args.results_dir}/fits.npz", **params)
