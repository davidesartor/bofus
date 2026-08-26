"""Build a standalone interactive HTML visualization from a run npz."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from targets import Brachistochrone, Projection
from vlse import GramacyLee

TEMPLATE = Path(__file__).parent / "viz_template.html"

LABELS = dict(
    gramacylee_projection=dict(
        TITLE="Gramacy-Lee Projection Run",
        BLURB=(
            'Bayesian optimization in function space: each query is an RBF mixture '
            '<span class="mono">f</span> with <span id="mCount">8</span> basis points, '
            'and the objective is the L2 distance between <span class="mono">f</span> '
            'and the Gramacy-Lee profile on the unit interval. Scrub iterations to '
            'watch the search shape the candidate toward the target.'
        ),
        METRIC="L2 distance",
        METRIC_SHORT="L2",
        TARGET_LABEL="Gramacy-Lee target",
        DOMAIN_LABEL="QUADRATURE DOMAIN",
    ),
    brachistochrone=dict(
        TITLE="Brachistochrone Descent Run",
        BLURB=(
            'Bayesian optimization in function space: each query is an RBF mixture '
            '<span class="mono">f</span> with <span id="mCount">8</span> basis points '
            'defining a descent curve from (-3, 1) to (0, 0), and the objective is the '
            'descent time in excess of the optimal cycloid. Scrub iterations to watch '
            'the search bend the candidate toward the brachistochrone.'
        ),
        METRIC="time gap (s)",
        METRIC_SHORT="gap",
        TARGET_LABEL="optimal cycloid",
        DOMAIN_LABEL="DESCENT DOMAIN",
    ),
)


def mixture_curves(l, x, a, tn):
    """Evaluate each candidate mixture on the normalized grid."""
    d2 = jnp.sum(jnp.square(tn[:, None, None, None, None] - x) / l, axis=-1)
    return jnp.sum(a * jnp.exp(-0.5 * d2), axis=-1).squeeze(-1).T  # (n, grid)


def gramacylee_data(l, x, a, grid_points):
    t = jnp.linspace(-0.5, 1.5, grid_points)
    curves = mixture_curves(l, x, a, t)
    target = Projection(GramacyLee(normalized=True))
    profile = jax.vmap(GramacyLee(normalized=True))(t[:, None])
    profile = (profile - target.shift) / target.scale
    profile = [float(v) if 0.0 <= ti <= 1.0 else None for v, ti in zip(profile, t)]
    return t, [0.0, 1.0], profile, curves, x.squeeze((1, 3)), l.squeeze((1, 3))


def brachistochrone_data(l, x, a, grid_points):
    target = Brachistochrone()
    (x0, y0), (x1, y1) = target.initial_position, target.final_position
    span = x1 - x0
    t = jnp.linspace(x0 - 0.05 * span, x1 + 0.05 * span, grid_points)
    tn = (t - x0) / span

    # endpoint-pinning offset and straight-line clip, as in the target itself
    f_vals = mixture_curves(l, x, a, tn)
    f_ends = mixture_curves(l, x, a, jnp.array([0.0, 1.0]))
    offset = (1 - tn) * (y0 - f_ends[:, :1]) + tn * (y1 - f_ends[:, 1:])
    curves = jnp.minimum(f_vals + offset, y0 + (y1 - y0) * tn)

    cycloid, _ = target.find_brachistochrone()
    profile = [float(cycloid(ti)) if x0 <= ti <= x1 else None for ti in np.asarray(t)]
    centers = x0 + span * x.squeeze((1, 3))
    return t, [x0, x1], profile, curves, centers, span**2 * l.squeeze((1, 3))


def doubling_stages(n, initial_acquisitions):
    """Stage regions [m, start, end) matching run.py's doubling schedule."""
    stages, s0, m = [], initial_acquisitions, 1
    while s0 < n:
        s1 = min(2 * s0, n)
        stages.append([m, s0, s1])
        s0, m = s1, 2 * m
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path")
    parser.add_argument("out_path")
    parser.add_argument("--target", choices=list(LABELS.keys()))
    parser.add_argument("--initial_acquisitions", type=int, default=8)
    parser.add_argument("--grid_points", type=int, default=384)
    args = parser.parse_args()

    # infer target, kernel, seed, and m from the results path convention
    target = args.target or next(k for k in LABELS if k in args.npz_path)
    kernel = re.search(r"(rbf|matern\d\d)", args.npz_path)
    seed_m = re.search(r"seed_(\d+)_m_(\d+)", args.npz_path)

    data = np.load(args.npz_path)
    l, x, a, y = data["l"], data["x"], data["a"], data["y"]
    n = len(y)

    builder = dict(
        gramacylee_projection=gramacylee_data, brachistochrone=brachistochrone_data
    )[target]
    t, domain, profile, curves, centers, lengthscales = builder(
        l, x, a, args.grid_points
    )

    payload = json.dumps(
        dict(
            t=t.tolist(),
            domain=domain,
            profile=profile,
            curves=np.asarray(curves).round(5).tolist(),
            y=y.tolist(),
            centers=np.asarray(centers).tolist(),
            lengthscales=np.asarray(lengthscales).tolist(),
            amplitudes=a.squeeze(1).tolist(),
            n=n,
        )
    )
    meta = json.dumps(
        dict(
            target=target,
            kernel=kernel.group(1) if kernel else None,
            seed=int(seed_m.group(1)) if seed_m else None,
            m=int(seed_m.group(2)) if seed_m else None,
            initial_acquisitions=args.initial_acquisitions,
            stages=doubling_stages(n, args.initial_acquisitions),
        )
    )

    html = TEMPLATE.read_text()
    for key, value in LABELS[target].items():
        html = html.replace("{{" + key + "}}", value)
    html = html.replace("/*__DATA__*/;", payload + ";")
    html = html.replace("/*__META__*/;", meta + ";")
    Path(args.out_path).write_text(html)
    print(f"wrote {args.out_path}: {n} iterations, best {np.nanmin(y):.5f}")


if __name__ == "__main__":
    main()
