"""Export a gramacylee_projection run npz to JSON for the visualization artifact."""

import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp

from vlse import GramacyLee


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path")
    parser.add_argument("out_path")
    parser.add_argument("--grid_points", type=int, default=256)
    parser.add_argument("--t_min", type=float, default=-0.5)
    parser.add_argument("--t_max", type=float, default=1.5)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    l, x, a, y = data["l"], data["x"], data["a"], data["y"]  # (n,k,m,d) (n,k,m) (n,)
    n = y.shape[0]

    t = jnp.linspace(args.t_min, args.t_max, args.grid_points)

    # evaluate each candidate mixture on the grid: sum_j a_j exp(-0.5 (t-x_j)^2 / l_j)
    d2 = jnp.sum(jnp.square(t[:, None, None, None, None] - x) / l, axis=-1)
    curves = jnp.sum(a * jnp.exp(-0.5 * d2), axis=-1).squeeze(-1).T  # (n, grid)

    # profile only on the quadrature domain [0, 1]; null elsewhere
    profile = jax.vmap(GramacyLee(normalized=True))(t[:, None])
    profile = [
        float(v) if 0.0 <= ti <= 1.0 else None for v, ti in zip(profile, t)
    ]

    out = dict(
        t=t.tolist(),
        domain=[0.0, 1.0],
        profile=profile,
        curves=np.asarray(curves).round(5).tolist(),
        y=y.tolist(),
        centers=x.squeeze((1, 3)).tolist(),
        lengthscales=l.squeeze((1, 3)).tolist(),
        amplitudes=a.squeeze(1).tolist(),
        n=n,
    )
    with open(args.out_path, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out_path}: {n} iterations, best {np.nanmin(y):.5f}")


if __name__ == "__main__":
    main()
