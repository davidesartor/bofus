"""Build an interactive Plotly dashboard of the m-bump RBF fits saved by fit_paper_references.py."""

import argparse
import json
from pathlib import Path

import numpy as np

from targets.whip import Whip

TEMPLATE = Path(__file__).parent / "templates" / "sparsity_dashboard.html"
REFERENCES = [f"whip{i}" for i in range(1, 7)] + ["whip2d"]
JOINTS_3D = ["shoulder flexion", "shoulder abduction", "shoulder rotation", "elbow"]
JOINTS_2D = ["shoulder", "elbow"]
SAMPLES = 251


def reference_payload(name: str, rows: list[dict]) -> dict:
    """The paper's reference in the BO's normalized coordinates, subsampled, plus its hit time."""
    whip = Whip(planar=True) if name == "whip2d" else Whip(target=int(name[-1]))
    ts, q0, _ = whip.paper_reference(1 if name == "whip2d" else int(name[-1]))
    fstar = np.asarray((q0 - whip.reference_center) / whip.reference_amplitude, dtype=float)
    sub = np.linspace(0, len(ts) - 1, SAMPLES).astype(int)
    first = next(r for r in rows if r["reference"] == name)
    return dict(
        joints=JOINTS_2D if whip.planar else JOINTS_3D,
        horizon=whip.horizon,
        t=(np.asarray(ts, dtype=float)[sub] / whip.horizon).round(4).tolist(),
        f=fstar[sub].round(4).tolist(),
        t_hit=first["t_hit"],
        cost_paper=first["cost_paper"],
        target=np.asarray(whip.target, dtype=float).round(3).tolist(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/paper_sparsity")
    parser.add_argument("--out", default="results/paper_sparsity/sparsity_dashboard.html")
    args = parser.parse_args()
    results = Path(args.results_dir)
    rows = json.load(open(results / "fits.json"))
    params = np.load(results / "fits.npz")

    # fits keyed by reference/l_floor/m: the bump parameters (k, 3m) and the scored metrics
    fits = {}
    for r in rows:
        key = f"{r['reference']}/l_{r['l_floor']}/m_{r['m']}"
        fits[key] = dict(
            params=np.asarray(params[key], dtype=float).round(6).tolist(),
            **{k: r[k] for k in ("rmse_prehit", "rmse_full", "cost", "min_l")},
        )
    payload = dict(
        references={n: reference_payload(n, rows) for n in REFERENCES if any(r["reference"] == n for r in rows)},
        l_floors=sorted({r["l_floor"] for r in rows}, reverse=True),
        ms=sorted({r["m"] for r in rows}),
        fits=fits,
    )
    html = TEMPLATE.read_text().replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) / 1e3:.0f} KB)")
