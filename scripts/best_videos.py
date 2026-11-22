"""Render each gym problem's best policy, open or closed loop, at full resolution next to the dashboard."""

import argparse
import functools
import glob
import json
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

# one thread per video worker
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
import numpy as np


def best_runs(results_dir: str) -> dict[str, dict]:
    """Per env id, the run file and index of the lowest objective across open and closed loop."""
    best = {}
    for path in glob.glob(f"{results_dir}/*loop/*/*/*/seed_*.npz"):
        name, method, profile, file = path.split(os.sep)[-4:]
        y = np.load(path)["y"]
        if not np.isfinite(y).any():
            continue
        index = int(np.argmin(np.where(np.isfinite(y), y, np.inf)))
        env_id = name.removesuffix("-open-loop")
        if env_id not in best or y[index] < best[env_id]["y"]:
            best[env_id] = dict(
                env=env_id,
                target=name,
                loop="open" if name.endswith("-open-loop") else "closed",
                method=method,
                profile=profile,
                seed=int(file.split("_")[1]),
                m=int(file.removesuffix(".npz").split("_")[-1]),
                index=index,
                y=float(y[index]),
                path=path,
            )
    return best


def full_resolution(gym_defaults: bool):
    """Record every frame at 1080p; optionally restore gym's horizons and observations."""
    import gymnasium as gym
    from targets import gymnasium as targets

    if gym_defaults:
        targets.HORIZONS.update(
            {k: gym.spec(k).max_episode_steps for k in targets.HORIZONS}
        )
        targets.OBSERVATION_KWARGS.clear()

    def gym_make(env_id: str, make=targets.gym_make, **kwargs) -> gym.Env:
        if kwargs.get("render_mode") and "mujoco" in gym.spec(env_id).entry_point:
            kwargs |= dict(width=1920, height=1080)
        return make(env_id, **kwargs)

    targets.gym_make = gym_make
    targets.VideoRecorder = functools.partial(
        targets.VideoRecorder, frame_stride=1, pixel_stride=1
    )


def render(run: dict, out: str, gym_defaults: bool) -> float:
    """Record one rollout of the run's best candidate to out; return its objective."""
    full_resolution(gym_defaults)
    import jax.numpy as jnp
    from run import target_fns
    from bofus import rkhs

    data = np.load(run["path"])
    f = rkhs.RBFMixture(*(jnp.asarray(data[k][run["index"]]) for k in "lxa"))
    with tempfile.TemporaryDirectory() as tmp:
        target = target_fns[run["target"]](n_rollouts=1, video_dir=tmp)
        value = float(target(f, x_range=(-1.0, 1.0), periodic=getattr(target, "periodic", None)))
        shutil.move(sorted(glob.glob(f"{tmp}/*.mp4"))[0], out)
    return value


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Best Policies</title>
<style>
  :root { color-scheme: light; --bg: #f4f4f2; --card: #fcfcfb; --border: #e2e1dc; --text: #0b0b0b; --muted: #7a7974; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    color-scheme: dark; --bg: #121211; --card: #1a1a19; --border: #2e2e2c; --text: #f0efec; --muted: #8f8e86; } }
  :root[data-theme="dark"] { color-scheme: dark; --bg: #121211; --card: #1a1a19; --border: #2e2e2c; --text: #f0efec; --muted: #8f8e86; }
  body { margin: 0; padding: 16px; background: var(--bg); color: var(--text); font: 13px system-ui, sans-serif; }
  main { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr)); gap: 12px; }
  figure { margin: 0; background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #000; object-fit: contain; }
  figcaption { padding: 8px 10px; }
  b { font-family: ui-monospace, monospace; font-weight: 500; }
  span { color: var(--muted); }
  .empty { padding: 24px; color: var(--muted); }
</style>
</head>
<body>
<main id="grid"></main>
<script>
const DATA = /*DATA*/{};
const grid = document.getElementById("grid");
for (const run of DATA.runs) {
  grid.insertAdjacentHTML("beforeend", `<figure>
    <video data-src="${run.video}" autoplay muted loop playsinline controls></video>
    <figcaption><b>${run.env}</b><br><span>${run.loop} loop · ${run.method} · seed ${run.seed} ·
      acquisition ${run.index + 1} · objective ${run.y.toPrecision(4)}</span></figcaption></figure>`);
}
// VS Code's preview frame cannot load local media, so there the outer frame fetches each video as a blob
const inPreview = !/^(file|https?):$/.test(location.protocol) && DATA.root;
for (const video of grid.querySelectorAll("video[data-src]")) {
  const rel = video.dataset.src;
  const failed = error => video.outerHTML = `<div class="empty">video failed: ${error} (${rel})</div>`;
  if (!inPreview) { video.src = rel; video.onerror = () => failed(video.error?.message); continue; }
  window.parent.fetch(`https://file+.vscode-resource.vscode-cdn.net${encodeURI(DATA.root)}/${encodeURI(rel)}`)
    .then(r => r.ok ? r.blob() : Promise.reject(new Error(`HTTP ${r.status}`)))
    .then(blob => video.src = URL.createObjectURL(new Blob([blob], { type: "video/mp4" })), failed);
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/sweep")
    parser.add_argument(
        "--gym_defaults",
        action="store_true",
        help="gym's horizons and observations, for runs made before HORIZONS",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    runs = sorted(best_runs(args.results_dir).values(), key=lambda run: run["env"])
    os.makedirs(os.path.join(args.results_dir, "best_videos"), exist_ok=True)
    for run in runs:
        run["video"] = f"best_videos/{run['env']}.mp4"

    # spawn, since forking a process with jax initialized can deadlock
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(min(args.jobs, len(runs)), mp_context=context) as pool:
        futures = {
            run["env"]: pool.submit(
                render,
                run,
                os.path.join(args.results_dir, run["video"]),
                args.gym_defaults,
            )
            for run in runs
        }
        for run in runs:
            run["rollout"] = futures[run["env"]].result()
            print(f"{run['env']}: logged {run['y']:.4g}, rollout {run['rollout']:.4g}")

    data = dict(runs=runs, root=os.path.abspath(args.results_dir))
    out = os.path.join(args.results_dir, "best_videos.html")
    with open(out, "w") as f:
        f.write(PAGE.replace("/*DATA*/{}", json.dumps(data)))
    print(f"{len(runs)} videos -> {out}")
