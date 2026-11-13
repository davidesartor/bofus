"""Build a standalone HTML dashboard of convergence and per-problem detail views."""

import argparse
import contextlib
import glob
import json
import multiprocessing
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import imageio_ffmpeg

# one thread per video worker
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from tqdm import tqdm

# run turns on float64 for the experiments; plotting only needs float32
from run import target_fns
from bofus import rkhs

jax.config.update("jax_enable_x64", False)


def group_of(target: str) -> str:
    """Tab a target belongs to."""
    if target.endswith("_projection"):
        return "projection"
    if target.endswith("_ridge"):
        return "ridge"
    return "open" if target.endswith("-open-loop") else "closed"


def unit_grid(d: int, n: int) -> Float[Array, "p d"]:
    """Tensor grid of n cell centers per axis on the unit box."""
    axis = (jnp.arange(n) + 0.5) / n
    return jnp.stack(jnp.meshgrid(*[axis] * d, indexing="ij"), axis=-1).reshape(-1, d)


def evaluate(
    fs: rkhs.RBFMixture, points: Float[Array, "p d"], periodic=None
) -> Float[Array, "n o p"]:
    """All candidates at all points."""
    return jnp.moveaxis(jax.vmap(lambda x: fs(x, periodic=periodic))(points), 0, -1)


def rounded(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), 4).tolist()


def incumbent_indices(y: np.ndarray) -> list[int]:
    """Acquisitions that improved the best value so far."""
    best, indices = np.inf, []
    for i, v in enumerate(y):
        if np.isfinite(v) and v < best:
            best = v
            indices.append(i)
    return indices


def subsample(target, count: int = 400) -> np.ndarray:
    """Evenly strided indices into the target's quadrature points."""
    return np.linspace(0, len(target.x) - 1, count).astype(int)


def target_view(name: str, target) -> dict:
    """Static per-target data: the reference the candidates are drawn against."""
    group, d = group_of(name), target.d
    if group == "projection":
        if d <= 2:
            points = unit_grid(d, 200 if d == 1 else 40)
            values = (jax.vmap(target.profile)(points) - target.shift) / target.scale
            return dict(
                kind="curve" if d == 1 else "heat",
                n=len(points) if d == 1 else 40,
                target=rounded(values),
            )
        return dict(kind="scatter", target=rounded(target.y[subsample(target)]))
    if group == "ridge":
        if d <= 2:
            points = unit_grid(d, 200 if d == 1 else 40)
            return dict(
                kind="curve" if d == 1 else "heat",
                n=len(points) if d == 1 else 40,
                target=rounded(jax.vmap(target.profile)(points)),
            )
        return dict(kind="points")
    if group == "open":
        horizon = gym.spec(target.env_id).max_episode_steps
        steps = np.unique(np.linspace(0, horizon - 1, min(horizon, 200)).astype(int))
        return dict(kind="actions", t=steps.tolist(), horizon=horizon, k=target.k)
    return dict(kind="video")


def run_detail(name: str, target, fs: rkhs.RBFMixture, incumbents: list[int]) -> dict:
    """Per-run data indexed like incumbents (or over all candidates for ridge)."""
    group, d = group_of(name), target.d
    if group == "projection":
        if d <= 2:
            points = unit_grid(d, 200 if d == 1 else 40)
        else:
            points = target.x[subsample(target)]
        chosen = jax.tree.map(lambda z: z[np.array(incumbents)], fs)
        return dict(f=rounded(jnp.tanh(evaluate(chosen, 2 * points - 1)[:, 0])))
    if group == "ridge":
        # each candidate's position in the profile's domain: sigmoid(b + int w tanh f)
        fx = jnp.tanh(evaluate(fs, 2 * target.x - 1)[:, 0])
        coords = jax.nn.sigmoid(
            target.b + jnp.einsum("dp,np->nd", target.w, fx) / target.x.shape[0]
        )
        return dict(coords=rounded(coords))
    if group == "open":
        horizon = gym.spec(target.env_id).max_episode_steps
        steps = np.unique(np.linspace(0, horizon - 1, min(horizon, 200)).astype(int))
        t = jnp.asarray(2 * steps / (horizon - 1) - 1)[:, None]
        chosen = jax.tree.map(lambda z: z[np.array(incumbents)], fs)
        return dict(u=rounded(jnp.tanh(evaluate(chosen, t))))
    return {}


def render_video(name: str, f: rkhs.RBFMixture, path: str) -> bool:
    """Record one rollout of candidate f to path."""
    with tempfile.TemporaryDirectory() as tmp:
        target = target_fns[name](n_rollouts=1, video_dir=tmp)
        try:
            target(f, periodic=getattr(target, "periodic", None))
        except Exception as error:
            tqdm.write(f"video failed for {path}: {error}")
            return False
        clips = sorted(glob.glob(f"{tmp}/*.mp4"))
        if not clips:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        to_webm(clips[0], path)
    return True


def to_webm(source: str, path: str):
    """Re-encode to VP9 webm, since VS Code's preview cannot decode H.264."""
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            source,
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "36",
            "-b:v",
            "0",
            "-an",
            path,
        ],
        check=True,
    )


def silence_stderr():
    """Mute C-level stderr, where MuJoCo prints its deprecation warnings."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)


@contextlib.contextmanager
def muted_stderr():
    """Silence stderr for a block, then restore it for the progress bars."""
    saved = os.dup(2)
    silence_stderr()
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def render_videos(pending: list[tuple], jobs: int) -> list[bool]:
    """Render (name, f, path) rollouts on a pool of single-threaded processes."""
    if not pending:
        return []
    # spawn, since forking a process with jax initialized can deadlock
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        min(jobs, len(pending)), mp_context=context, initializer=silence_stderr
    ) as pool:
        names, fs, paths = zip(*pending)
        results = pool.map(render_video, names, fs, paths)
        return list(tqdm(results, total=len(pending), desc="videos", unit="video"))


def build(results_dir: str, videos: bool, jobs: int) -> dict:
    """Targets with their static views, and one record per checkpointed run."""
    targets, runs, pending = {}, [], []
    progress = tqdm(sorted(glob.glob(f"{results_dir}/*/*/*/seed_*.npz")), unit="run")
    for path in progress:
        name, method, profile, file = path.split(os.sep)[-4:]
        progress.set_description(f"{name} {method}")
        if name not in target_fns:
            continue
        if name not in targets:
            with muted_stderr():
                target = target_fns[name]()
            targets[name] = dict(
                target=target,
                group=group_of(name),
                d=target.d,
                k=target.k,
                view=target_view(name, target),
            )
        target = targets[name]["target"]

        data = np.load(path)
        fs = rkhs.RBFMixture(
            jnp.asarray(data["l"]), jnp.asarray(data["x"]), jnp.asarray(data["a"])
        )
        y = np.asarray(data["y"])
        optimum = float(data["optimum"])
        incumbents = incumbent_indices(y)
        record = dict(
            target=name,
            method=method,
            profile=profile,
            seed=int(file.split("_")[1]),
            y=[float(v) if np.isfinite(v) else None for v in y],
            optimum=None if np.isnan(optimum) else optimum,
            updated=os.path.getmtime(path),
            incumbents=incumbents,
            detail=run_detail(name, target, fs, incumbents) if incumbents else {},
        )

        # rollout video of the best candidate of control runs, next to the dashboard
        if videos and incumbents and group_of(name) in ("open", "closed"):
            best = incumbents[-1]
            relative = (
                f"videos/{name}/{method}/{profile}/seed_{record['seed']}_{best}.webm"
            )
            record["videos"] = {best: relative}
            path = os.path.join(results_dir, relative)
            if not os.path.exists(path):
                f = jax.tree.map(lambda z: np.asarray(z[best]), fs)
                pending.append((name, f, path, record))
        runs.append(record)

    # drop the videos whose rollout failed
    succeeded = render_videos([job[:3] for job in pending], jobs)
    for (*_, record), ok in zip(pending, succeeded):
        if not ok:
            record["videos"] = {}

    for entry in targets.values():
        del entry["target"]
    return dict(targets=targets, runs=runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/sweep")
    parser.add_argument(
        "--out", default=None, help="defaults to <results_dir>/dashboard.html"
    )
    parser.add_argument(
        "--videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render a rollout of each control run's best candidate",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    template_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(template_path) as f:
        template = f.read()

    data = build(args.results_dir, args.videos, args.jobs)
    out = args.out or os.path.join(args.results_dir, "dashboard.html")
    with open(out, "w") as f:
        f.write(template.replace("/*DATA*/{}", json.dumps(data)))
    print(f"{len(data['runs'])} runs -> {out}")
