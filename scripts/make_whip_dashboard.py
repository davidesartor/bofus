"""Build an interactive Plotly dashboard from the saved whip runs."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from tqdm import tqdm
from jaxtyping import Array, Float, Scalar

from bofus.rkhs import RBFMixture
from targets.whip import Whip, joint_positions

TEMPLATE = Path(__file__).parent / "templates" / "whip_dashboard.html"
CACHE_VERSION = 2
MODELS = (1, 2, 4)
TARGETS = range(1, 7)
# DIRECT-L iterations to converge, from the paper text (exact only for targets 3 and 5)
PAPER_ITERATIONS = {1: (100, "<100"), 2: (100, "<100"), 3: (39, "39"), 4: (250, "<250"), 5: (249, "249"), 6: (250, "<250")}


@eqx.filter_jit
def simulate_reference(
    whip: Whip, q0: Float[Array, "t 4"], dq0: Float[Array, "t 4"]
) -> tuple[Scalar, Float[Array, "48 n/2+1 3"]]:
    """Roll out a reference trajectory: its objective and subsampled joint positions."""
    _, qs, _ = whip.rollout(q0, dq0)
    frame_indices = jnp.linspace(0, whip.n_steps, 48).astype(int)
    frames = jax.vmap(joint_positions, in_axes=(0, None))(qs[frame_indices], whip.chain)
    return whip.min_distance(qs), frames


def animate_solution(whip: Whip, data: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    """Roll out one saved candidate for the physical animation."""
    f = RBFMixture(*(jnp.asarray(data[k][index]) for k in ("l", "x", "a")))
    _, q0, dq0 = whip.reference(f)
    _, frames = simulate_reference(whip, q0, dq0)
    return np.asarray(frames).round(4)


def paper_solution(whip: Whip, target: int) -> dict:
    """The paper's Table 1 min-jerk move for this target, scored with the same objective."""
    _, q0, dq0 = whip.paper_reference(target)
    cost, frames = simulate_reference(whip, q0, dq0)
    iterations, label = PAPER_ITERATIONS[target]
    return {
        "y": float(cost),
        "iterations": iterations,
        "iterationsLabel": label,
        "poses": np.asarray(frames).round(4).tolist(),
    }


def running_best_poses(
    whip: Whip, path: Path, cache_path: Path, progress: tqdm
) -> tuple[list[int], list]:
    """Animate every running-best candidate, reusing the cache when the run is unchanged."""
    data = np.load(path)
    pose_steps = np.flatnonzero(data["y"] == np.minimum.accumulate(data["y"])).tolist()
    signature = np.array([CACHE_VERSION, path.stat().st_mtime_ns, path.stat().st_size])
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["source_signature"], signature):
            progress.update(len(pose_steps))
            return cached["pose_steps"].astype(int).tolist(), cached["poses"].tolist()

    poses = []
    for step in pose_steps:
        poses.append(animate_solution(whip, data, step).tolist())
        progress.update()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        source_signature=signature,
        pose_steps=np.asarray(pose_steps, dtype=np.int32),
        poses=np.asarray(poses, dtype=np.float32),
    )
    return pose_steps, poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out", default="plots/whip_dashboard.html")
    parser.add_argument("--cache_dir", default="plots/.cache/whip_dashboard")
    args = parser.parse_args()

    paths = {
        (model, target): Path(
            f"{args.results_dir}/whip{target}/ours_adaptive/matern52/seed_0_m_{model}.npz"
        )
        for model in MODELS
        for target in TARGETS
    }
    total_candidates = sum(
        int(np.count_nonzero(y == np.minimum.accumulate(y)))
        for path in paths.values()
        if path.exists()
        for y in [np.load(path)["y"]]
    )

    payload = {model: {} for model in MODELS}
    payload["tipNodes"] = Whip().n_tip_nodes
    payload["paper"] = {
        target: paper_solution(Whip(target=target), target)
        for target in tqdm(TARGETS, desc="Simulating paper references", unit="target")
    }
    progress = tqdm(total=total_candidates, desc="Simulating whip candidates", unit="candidate")
    for (model, target), path in paths.items():
        if not path.exists():
            payload[model][target] = None
            continue
        progress.set_postfix(model=f"m={model}", target=f"whip{target}")
        whip = Whip(target=target)
        cache_path = Path(args.cache_dir) / f"m{model}_whip{target}.npz"
        pose_steps, poses = running_best_poses(whip, path, cache_path, progress)
        y = np.load(path)["y"].astype(float)
        payload[model][target] = {
            "y": y.tolist(),
            "best": np.minimum.accumulate(y).tolist(),
            "target": np.asarray(whip.target).round(4).tolist(),
            "poseSteps": pose_steps,
            "poses": poses,
        }
    progress.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text().replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
