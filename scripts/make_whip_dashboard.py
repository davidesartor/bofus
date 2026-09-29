"""Build an interactive Plotly dashboard from the saved whip runs, complete or in flight."""

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
CACHE_VERSION = 5
ANIMATION_FPS = 24
# rollouts animated per run: running-best improvements can be dense, and each is ~100 KB of JSON
MAX_POSE_STEPS = 16
MODELS = (1, 2, 4)
TASKS = {"whip2d": "2D", **{f"whip3d_{i}": f"3D target {i}" for i in range(1, 7)}}
# DIRECT-L iterations to converge, from the paper text (exact only for targets 3 and 5)
PAPER_ITERATIONS = {
    1: (100, "<100"),
    2: (100, "<100"),
    3: (39, "39"),
    4: (250, "<250"),
    5: (249, "249"),
    6: (250, "<250"),
}


def make_whip(task: str) -> Whip:
    return Whip(planar=True) if task == "whip2d" else Whip(target=int(task[-1]))


@eqx.filter_jit
def simulate_reference(
    whip: Whip, q0: Float[Array, "t k"], dq0: Float[Array, "t k"]
) -> tuple[Scalar, Float[Array, "f n/2+1 3"]]:
    """Roll out a reference trajectory: its objective and joint positions at ANIMATION_FPS."""
    _, qs, _ = whip.rollout(q0, dq0)
    n_frames = round(whip.horizon * ANIMATION_FPS) + 1
    frame_indices = jnp.linspace(0, whip.n_steps, n_frames).astype(int)
    frames = jax.vmap(joint_positions, in_axes=(0, None))(qs[frame_indices], whip.chain)
    return whip.min_distance(qs), frames


def animate_solution(whip: Whip, data: np.lib.npyio.NpzFile, index: int) -> np.ndarray:
    """Roll out one saved candidate for the physical animation."""
    f = RBFMixture(*(jnp.asarray(data[k][index]) for k in ("l", "x", "a")))
    _, q0, dq0 = whip.reference(f)
    _, frames = simulate_reference(whip, q0, dq0)
    return np.asarray(frames, dtype=float).round(3)


def paper_solution(whip: Whip, task: str, direct_path: Path) -> dict:
    """The paper's own move for this task, scored with the same objective.

    When our port of the paper's DIRECT-L run (scripts/run_whip_direct.py) has been
    saved, its evaluation trace replaces the iteration count quoted from the paper.
    """
    planar = task == "whip2d"
    _, q0, dq0 = whip.paper_reference(1 if planar else int(task[-1]))
    cost, frames = simulate_reference(whip, q0, dq0)
    iterations, label = (None, "n/a") if planar else PAPER_ITERATIONS[int(task[-1])]
    solution = {
        "y": float(cost),
        "iterations": iterations,
        "iterationsLabel": label,
        "poses": np.asarray(frames, dtype=float).round(3).tolist(),
    }
    if direct_path.exists():
        y = np.load(direct_path)["y"].astype(float)
        solution["directBest"] = np.minimum.accumulate(y).tolist()
    return solution


def running_best_poses(
    whip: Whip, path: Path, cache_path: Path, progress: tqdm
) -> tuple[list[int], list]:
    """Animate up to MAX_POSE_STEPS running-best candidates, reusing the cache when the run is unchanged."""
    data = np.load(path)
    improvements = np.flatnonzero(data["y"] == np.minimum.accumulate(data["y"]))
    keep = np.unique(np.linspace(0, len(improvements) - 1, MAX_POSE_STEPS).round().astype(int))
    pose_steps = improvements[keep].tolist()
    signature = np.array([CACHE_VERSION, path.stat().st_mtime_ns, path.stat().st_size])
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["source_signature"], signature):
            progress.update(len(pose_steps))
            return cached["pose_steps"].astype(int).tolist(), cached["poses"].astype(float).round(3).tolist()

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


def median_running_best(bests: list[np.ndarray]) -> list[float]:
    """Median across seeds at every step, over the seeds that have reached it."""
    width = max(len(b) for b in bests)
    padded = np.full((len(bests), width), np.nan)
    for row, b in zip(padded, bests):
        row[: len(b)] = b
    return np.nanmedian(padded, axis=0).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/icra")
    parser.add_argument("--method", default="rbf_ei")
    parser.add_argument(
        "--budget", type=int, default=256, help="Acquisitions a finished run has."
    )
    parser.add_argument("--out", default="plots/whip_dashboard.html")
    parser.add_argument("--cache_dir", default="plots/.cache/whip_dashboard")
    args = parser.parse_args()

    # every saved run, keyed by task and basis size, in seed order
    runs = {
        task: {
            model: sorted(
                Path(f"{args.results_dir}/{task}/{args.method}/matern52").glob(
                    f"seed_*_m_{model}.npz"
                ),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            for model in MODELS
        }
        for task in TASKS
    }
    payload = {
        "tipNodes": Whip().n_tip_nodes,
        "budget": args.budget,
        "method": args.method,
        "tasks": {},
        "runs": {task: {} for task in TASKS},
    }

    progress = tqdm(list(TASKS), desc="Simulating whip runs", unit="task")
    for task in progress:
        whip = make_whip(task)
        payload["tasks"][task] = {
            "label": TASKS[task],
            "planar": whip.planar,
            "target": np.asarray(whip.target).round(4).tolist(),
            "hitRadius": whip.hit_radius,
            "horizon": whip.horizon,
            "paper": paper_solution(
                whip, task, Path(f"{args.results_dir}/{task}/minjerk_direct/run.npz")
            ),
        }
        for model in MODELS:
            paths = runs[task][model]
            if not paths:
                continue
            seeds = {}
            for path in paths:
                y = np.load(path)["y"].astype(float)
                seeds[int(path.stem.split("_")[1])] = {
                    "y": y.tolist(),
                    "best": np.minimum.accumulate(y).tolist(),
                    "done": len(y) >= args.budget,
                }
            # the animation follows the seed whose running best is lowest so far
            animated = min(seeds, key=lambda s: seeds[s]["best"][-1])
            path = next(p for p in paths if int(p.stem.split("_")[1]) == animated)
            progress.set_postfix(model=f"m={model}", seed=animated)
            cache_path = Path(args.cache_dir) / f"{task}_m{model}_seed{animated}.npz"
            pose_steps, poses = running_best_poses(
                whip, path, cache_path, tqdm(disable=True)
            )
            payload["runs"][task][model] = {
                "seeds": seeds,
                "median": median_running_best(
                    [np.asarray(s["best"]) for s in seeds.values()]
                ),
                "animated": animated,
                "poseSteps": pose_steps,
                "poses": poses,
            }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text().replace(
        "/*__DATA__*/", json.dumps(payload, separators=(",", ":"))
    )
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
