"""Build a standalone HTML dashboard of convergence and per-problem detail views."""

import argparse
import contextlib
import glob
import json
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym

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
    return jnp.moveaxis(jax.vmap(lambda x: fs(x, (-1.0, 1.0), periodic))(points), 0, -1)


def rounded(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), 2).tolist()


def incumbent_indices(y: np.ndarray) -> list[int]:
    """Acquisitions that improved the best value so far."""
    best, indices = np.inf, []
    for i, v in enumerate(y):
        if np.isfinite(v) and v < best:
            best = v
            indices.append(i)
    return indices


def subsample(target, count: int = 200) -> np.ndarray:
    """Evenly strided indices into the target's quadrature points."""
    return np.linspace(0, len(target.x) - 1, count).astype(int)


def target_view(name: str, target) -> dict:
    """Static per-target data: the reference the candidates are drawn against."""
    group, d = group_of(name), target.d
    if group == "projection":
        if d <= 2:
            points = unit_grid(d, 100 if d == 1 else 28)
            values = (jax.vmap(target.profile)(points) - target.shift) / target.scale
            return dict(
                kind="curve" if d == 1 else "heat",
                n=len(points) if d == 1 else 28,
                target=rounded(values),
            )
        return dict(kind="scatter", target=rounded(target.y[subsample(target)]))
    if group == "ridge":
        if d <= 2:
            points = unit_grid(d, 100 if d == 1 else 28)
            return dict(
                kind="curve" if d == 1 else "heat",
                n=len(points) if d == 1 else 28,
                target=rounded(jax.vmap(target.profile)(points)),
            )
        return dict(kind="points")
    if group == "open":
        horizon = gym.spec(target.env_id).max_episode_steps
        steps = np.unique(np.linspace(0, horizon - 1, min(horizon, 50)).astype(int))
        return dict(kind="actions", t=steps.tolist(), horizon=horizon, k=target.k)
    return dict(kind="video")


def run_detail(name: str, target, fs: rkhs.RBFMixture, incumbents: list[int]) -> dict:
    """Per-run data indexed like incumbents (or over all candidates for ridge)."""
    group, d = group_of(name), target.d
    if group == "projection":
        if d <= 2:
            points = unit_grid(d, 100 if d == 1 else 28)
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
        steps = np.unique(np.linspace(0, horizon - 1, min(horizon, 50)).astype(int))
        t = jnp.asarray(2 * steps / (horizon - 1) - 1)[:, None]
        chosen = jax.tree.map(lambda z: z[np.array(incumbents)], fs)
        return dict(u=rounded(jnp.tanh(evaluate(chosen, t))))
    return {}


def render_video(name: str, f: rkhs.RBFMixture, path: str) -> bool:
    """Record one rollout of candidate f to path."""
    with tempfile.TemporaryDirectory() as tmp:
        target = target_fns[name](n_rollouts=1, video_dir=tmp)
        try:
            target(f, x_range=(-1.0, 1.0), periodic=getattr(target, "periodic", None))
        except Exception as error:
            tqdm.write(f"video failed for {path}: {error}")
            return False
        clips = sorted(glob.glob(f"{tmp}/*.mp4"))
        if not clips:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.move(clips[0], path)
    return True


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
    progress = tqdm(sorted(glob.glob(f"{results_dir}/*/*/*/*/seed_*.npz")), unit="run")
    for path in progress:
        name, method, profile, file = path.split(os.sep)[-4:]
        _, seed, _, m = file.removesuffix(".npz").split("_")
        method = f"{method}_m{m}"
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
        # detail only for the last few improvements, to keep the html loadable
        incumbents = incumbent_indices(y)[-8:]
        record = dict(
            target=name,
            method=method,
            profile=profile,
            seed=int(seed),
            y=[float(v) if np.isfinite(v) else None for v in y],
            optimum=None if np.isnan(optimum) else optimum,
            updated=os.path.getmtime(path),
            incumbents=incumbents,
            detail=run_detail(name, target, fs, incumbents) if incumbents else {},
        )

        # rollout video of the best candidate of control runs, next to the dashboard
        if videos and incumbents and group_of(name) in ("open", "closed"):
            best = incumbents[-1]
            relative = f"videos/{name}/{method}/{profile}/seed_{seed}_{best}.mp4"
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
    return dict(targets=targets, runs=runs, root=os.path.abspath(results_dir))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sweep Dashboard</title>
<style>
  :root {
    color-scheme: light;
    --surface-0: #f4f4f2; --surface-1: #fcfcfb; --border: #e2e1dc; --grid: #ecebe7; --axis: #b9b8b1;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100; --series-5: #e87ba4;
    --neutral-mid: #ebeae6; --reference: #52514e;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2c; --grid: #262624; --axis: #4a4a46;
      --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500; --series-5: #d55181;
      --neutral-mid: #2c2c2a; --reference: #c3c2b7;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #2e2e2c; --grid: #262624; --axis: #4a4a46;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500; --series-5: #d55181;
    --neutral-mid: #2c2c2a; --reference: #c3c2b7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 16px 48px; background: var(--surface-0); color: var(--text-primary);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 1280px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
  h2 { font-size: 15px; margin: 28px 0 10px; font-weight: 600; }
  h3 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
  .sub { color: var(--text-secondary); margin: 0 0 18px; }
  .meta { font-size: 12px; color: var(--text-muted); margin: 0 0 8px; min-height: 17px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .stat .label { color: var(--text-secondary); font-size: 12px; }
  .stat .value { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin: 24px 0 14px; overflow-x: auto; }
  .tab {
    border: 0; background: none; color: var(--text-secondary); font: inherit; font-weight: 600;
    padding: 8px 14px; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; white-space: nowrap;
  }
  .tab[aria-selected="true"] { color: var(--text-primary); border-bottom-color: var(--text-primary); }
  .tab:disabled { opacity: 0.4; cursor: default; }
  .tab small { color: var(--text-muted); font-weight: 400; margin-left: 4px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin-bottom: 12px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary);
    border-radius: 999px; padding: 3px 11px; font: inherit; font-size: 13px; cursor: pointer;
    display: inline-flex; align-items: center; gap: 6px;
  }
  .chip[aria-pressed="true"] { color: var(--text-primary); border-color: var(--text-secondary); font-weight: 600; }
  .chip:disabled { opacity: 0.4; cursor: default; }
  .swatch { display: inline-block; width: 12px; height: 3px; border-radius: 2px; }
  .label { color: var(--text-muted); font-size: 12px; font-weight: 400; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; color: var(--text-secondary); font-size: 12px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .panels { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); gap: 12px; }
  @media (max-width: 860px) { .panels { grid-template-columns: minmax(0, 1fr); } }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  svg text { fill: var(--text-muted); font-size: 10px; font-variant-numeric: tabular-nums; }
  .player { display: flex; align-items: center; gap: 10px; margin: 4px 0 10px; }
  .player button {
    border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary);
    border-radius: 6px; width: 34px; height: 28px; cursor: pointer; font-size: 13px;
  }
  .player input { flex: 1; min-width: 0; accent-color: var(--text-secondary); }
  .player output { font-size: 12px; color: var(--text-secondary); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .heats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
  .heats figure { margin: 0; }
  .heats figcaption { font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }
  canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border-radius: 4px; display: block; }
  .colorbar { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-muted); margin-top: 8px; }
  .colorbar i { flex: 0 0 140px; height: 8px; border-radius: 2px; }
  video { width: 100%; max-height: 360px; border-radius: 6px; background: #000; display: block; }
  .tooltip {
    position: fixed; pointer-events: none; z-index: 10; display: none; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 12px;
    box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); font-variant-numeric: tabular-nums; min-width: 140px;
  }
  .tooltip .r { display: flex; align-items: center; gap: 6px; }
  .tooltip b { margin-left: auto; padding-left: 10px; }
  .table-wrap { overflow-x: auto; padding: 0; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th { color: var(--text-secondary); font-weight: 600; }
  .num { text-align: right; }
  .empty { color: var(--text-muted); padding: 24px; text-align: center; }
</style>
</head>
<body>
<main>
  <h1>Sweep dashboard</h1>
  <p class="sub" id="sub"></p>
  <div class="stats" id="stats"></div>

  <div class="tabs" id="tabs" role="tablist"></div>
  <div class="row"><span class="label">Problem</span><div class="chips" id="problems"></div></div>
  <div class="row" id="atoms-row"><span class="label">Atoms</span><div class="chips" id="atoms"></div></div>

  <div class="panels">
    <section class="card">
      <h3>Convergence</h3>
      <p class="meta" id="conv-meta"></p>
      <div id="conv"></div>
      <div class="legend" id="legend" style="margin-top:8px"></div>
    </section>
    <section class="card">
      <h3>Best candidates so far</h3>
      <div class="row" style="margin:6px 0 4px">
        <div class="chips" id="methods"></div>
        <div class="chips" id="seeds"></div>
      </div>
      <div class="player">
        <button id="play" aria-label="Play">▶</button>
        <input type="range" id="slider" min="0" max="0" value="0" aria-label="Improvement">
        <output id="pos"></output>
      </div>
      <p class="meta" id="detail-meta"></p>
      <div id="detail"></div>
    </section>
  </div>

  <h2>Summary</h2>
  <div class="card table-wrap"><table id="table"></table></div>
</main>
<div class="tooltip" id="tooltip"></div>

<script>
const DATA = /*DATA*/{};
const RUNS = DATA.runs || [], TARGETS = DATA.targets || {};

const TABS = [["projection", "Projection"], ["ridge", "Ridge"], ["open", "Open-loop control"], ["closed", "Closed-loop control"]];
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
// methods are <distance>_..._m<atoms>: color encodes the atoms, dashes the L2 distance
const atomsOf = m => +m.match(/_m(\d+)$/)[1];
const methods = [...new Set(RUNS.map(r => r.method))].sort((a, b) => {
  const rank = m => m.includes("rkhs") ? 0 : m.includes("l2") ? 1 : 2;
  return rank(a) - rank(b) || atomsOf(a) - atomsOf(b) || a.localeCompare(b);
});
const atoms = [...new Set(methods.map(atomsOf))].sort((a, b) => a - b);
const colorOf = m => `var(--series-${atoms.indexOf(atomsOf(m)) + 1})`;
const dashOf = m => m.includes("l2") ? "5 3" : "none";
const methodLabel = m => m.replace(/^ours_/, "").replace(/_adaptive/, "").replace("rkhs", "RKHS").replace("l2", "L2").replace(/_m(\d+)$/, " m=$1").replace(/_/g, " ");
const lineKey = m => `<svg width="18" height="8" style="vertical-align:middle;margin-right:4px"><line x1="0" y1="4" x2="18" y2="4" stroke="${colorOf(m)}" stroke-width="2" stroke-dasharray="${dashOf(m)}"/></svg>`;
const problemLabel = t => t.replace(/-open-loop$/, "").replace(/_(projection|ridge)$/, "");
const fmt = v => v == null || !isFinite(v) ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(2) : (+v.toPrecision(4)).toString();
const median = xs => { const s = [...xs].sort((a, b) => a - b), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const bestSoFar = y => { let b = Infinity; return y.map(v => (b = v == null ? b : Math.min(b, v))); };
const el = id => document.getElementById(id);
const tooltip = el("tooltip");

// header
const lastUpdate = Math.max(0, ...RUNS.map(r => r.updated));
el("sub").textContent = RUNS.length
  ? `Lower is better: regret for synthetic targets, −mean return for control. Last checkpoint ${new Date(lastUpdate * 1000).toLocaleString()}.`
  : "No checkpointed runs yet.";
const budget = Math.max(0, ...RUNS.map(r => r.y.length));
el("stats").innerHTML = [
  ["Problems", Object.keys(TARGETS).length], ["Runs", RUNS.length],
  ["Runs at full budget", RUNS.filter(r => r.y.length >= budget).length],
  ["Acquisitions", RUNS.reduce((s, r) => s + r.y.length, 0).toLocaleString()],
].map(([l, v]) => `<div class="card stat"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");

// selection state
const state = { tab: null, target: null, method: methods[0], seed: 0, k: 0, playing: null, atoms: new Set(atoms) };
const shownMethods = () => methods.filter(m => state.atoms.has(atomsOf(m)));
const targetsIn = tab => Object.keys(TARGETS).filter(t => TARGETS[t].group === tab).sort((a, b) => TARGETS[a].d - TARGETS[b].d || a.localeCompare(b));
const runsOf = t => RUNS.filter(r => r.target === t);
const currentRun = () => RUNS.find(r => r.target === state.target && r.method === state.method && r.seed === state.seed);

function chip(label, pressed, data, disabled = false, swatch = null) {
  const sw = swatch ? `<i class="swatch" style="background:${swatch}"></i>` : "";
  return `<button class="chip" aria-pressed="${pressed}" ${disabled ? "disabled" : ""} ${data}>${sw}${label}</button>`;
}

function renderTabs() {
  el("tabs").innerHTML = TABS.map(([key, label]) => {
    const n = targetsIn(key).length;
    return `<button class="tab" role="tab" aria-selected="${key === state.tab}" data-tab="${key}" ${n ? "" : "disabled"}>${label}<small>${n}</small></button>`;
  }).join("");
}
el("tabs").addEventListener("click", e => {
  const tab = e.target.closest("[data-tab]")?.dataset.tab;
  if (tab && targetsIn(tab).length) selectTab(tab);
});
el("problems").addEventListener("click", e => {
  const t = e.target.closest("[data-t]")?.dataset.t;
  if (t) selectTarget(t);
});
el("methods").addEventListener("click", e => {
  const m = e.target.closest("[data-m]")?.dataset.m;
  if (m) { state.method = m; fixRun(); renderDetailControls(); renderDetail(true); }
});
// m chips toggle which atom counts are shown everywhere; at least one stays on
function renderAtoms() {
  el("atoms-row").style.display = atoms.length > 1 ? "" : "none";
  el("atoms").innerHTML = atoms.map(a => chip(`m=${a}`, state.atoms.has(a), `data-a="${a}"`, false, `var(--series-${atoms.indexOf(a) + 1})`)).join("");
}
el("atoms").addEventListener("click", e => {
  const a = e.target.closest("[data-a]")?.dataset.a;
  if (a === undefined) return;
  if (!state.atoms.delete(+a)) state.atoms.add(+a);
  if (!state.atoms.size) state.atoms.add(+a);
  renderAtoms(); fixRun(); renderConvergence(); renderDetailControls(); renderDetail(true); renderTable();
});
renderAtoms();
el("seeds").addEventListener("click", e => {
  const s = e.target.closest("[data-s]")?.dataset.s;
  if (s !== undefined) { state.seed = +s; fixRun(); renderDetailControls(); renderDetail(true); }
});

function selectTab(tab) {
  state.tab = tab;
  renderTabs();
  selectTarget(targetsIn(tab)[0]);
}
// the seed with the best objective, within the method whose median best objective is lowest
const finalBest = r => bestSoFar(r.y).at(-1) ?? Infinity;
function bestRun(t) {
  const runs = runsOf(t), score = m => median(runs.filter(r => r.method === m).map(finalBest));
  const method = [...new Set(runs.map(r => r.method))].map(m => [score(m), m]).sort((a, b) => a[0] - b[0])[0]?.[1];
  return runs.filter(r => r.method === method).sort((a, b) => finalBest(a) - finalBest(b))[0];
}
// every problem opens unfiltered, on its best method and seed
function selectTarget(t) {
  state.target = t;
  state.atoms = new Set(atoms);
  const best = bestRun(t);
  if (best) { state.method = best.method; state.seed = best.seed; }
  renderAtoms();
  renderTable();
  el("problems").innerHTML = targetsIn(state.tab).map(x => {
    const dims = `d=${TARGETS[x].d}${TARGETS[x].k > 1 ? `, k=${TARGETS[x].k}` : ""}`;
    return chip(`${problemLabel(x)} <span class="label">${dims}</span>`, x === t, `data-t="${x}"`);
  }).join("");
  fixRun();
  renderConvergence();
  renderDetailControls();
  renderDetail(true);
}
// keep method and seed on a run that exists for this target
function fixRun() {
  const shown = shownMethods(), runs = runsOf(state.target).filter(r => shown.includes(r.method));
  if (!runs.some(r => r.method === state.method)) state.method = runs[0]?.method;
  const seeds = runs.filter(r => r.method === state.method).map(r => r.seed);
  if (!seeds.includes(state.seed)) state.seed = Math.min(...seeds);
}
function renderDetailControls() {
  const runs = runsOf(state.target);
  el("methods").innerHTML = shownMethods().map(m => chip(methodLabel(m), m === state.method, `data-m="${m}"`, !runs.some(r => r.method === m), colorOf(m))).join("");
  const seeds = [...new Set(runs.map(r => r.seed))].sort((a, b) => a - b);
  el("seeds").innerHTML = seeds.map(s => chip(`seed ${s}`, s === state.seed, `data-s="${s}"`, !runs.some(r => r.seed === s && r.method === state.method))).join("");
}

// ---------- generic svg frame ----------
const niceTicks = (lo, hi, count) => {
  const span = hi - lo || Math.abs(hi) || 1, step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0)), step = [1, 2, 5, 10].map(k => k * mag).find(s => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
};
function frame({ W = 420, H = 240, xlo, xhi, ylo, yhi, log = false, yTicks, xLabel = "" }) {
  const M = { l: 48, r: 10, t: 8, b: xLabel ? 32 : 22 };
  const sx = v => M.l + (xhi > xlo ? (v - xlo) / (xhi - xlo) : 0.5) * (W - M.l - M.r);
  const ty = log ? v => Math.log10(Math.max(v, 1e-300)) : v => v;
  const sy = v => M.t + (1 - (ty(v) - ylo) / (yhi - ylo)) * (H - M.t - M.b);
  const yt = (yTicks || niceTicks(ylo, yhi, 4)).map(t => {
    const py = M.t + (1 - (t - ylo) / (yhi - ylo)) * (H - M.t - M.b);
    return `<line x1="${M.l}" x2="${W - M.r}" y1="${py}" y2="${py}" stroke="var(--grid)"/><text x="${M.l - 6}" y="${py + 3}" text-anchor="end">${log ? `1e${t}` : fmt(t)}</text>`;
  }).join("");
  const xt = niceTicks(xlo, xhi, 5).filter(t => t >= xlo && t <= xhi).map(t =>
    `<text x="${sx(t)}" y="${H - M.b + 14}" text-anchor="middle">${fmt(t)}</text>`).join("");
  const axes = `${yt}<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--axis)"/>${xt}` +
    (xLabel ? `<text x="${(M.l + W - M.r) / 2}" y="${H - 3}" text-anchor="middle">${xLabel}</text>` : "");
  const path = (xs, ys) => xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join("");
  return { W, H, M, sx, sy, axes, path };
}

// ---------- convergence ----------
function renderConvergence() {
  const runs = runsOf(state.target), log = runs[0]?.optimum != null;
  const series = shownMethods().map(method => {
    const curves = runs.filter(r => r.method === method).map(r => bestSoFar(r.y).map(v => log ? Math.max(v - r.optimum, 1e-12) : v));
    const n = Math.max(0, ...curves.map(c => c.length));
    const med = Array.from({ length: n }, (_, i) => median(curves.filter(c => c.length > i).map(c => c[i])));
    return { method, curves, med };
  }).filter(s => s.curves.length);
  const all = series.flatMap(s => s.curves.flat()).filter(isFinite);
  const n = Math.max(...series.map(s => s.med.length));
  el("conv-meta").textContent = `${log ? "simple regret, log scale" : "−mean return"} vs acquisition · thick = median, thin = seeds`;
  el("legend").innerHTML = series.map(s => `<span>${lineKey(s.method)}${methodLabel(s.method)} (${s.curves.length} seed${s.curves.length > 1 ? "s" : ""})</span>`).join("");
  if (!all.length) { el("conv").innerHTML = `<div class="empty">no data</div>`; return; }
  let ylo, yhi;
  if (log) { ylo = Math.floor(Math.log10(Math.min(...all))); yhi = Math.max(Math.ceil(Math.log10(Math.max(...all))), ylo + 1); }
  else { const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo || 1) * 0.05; ylo = lo - pad; yhi = hi + pad; }
  const yTicks = log ? Array.from({ length: yhi - ylo + 1 }, (_, k) => ylo + k).filter((_, k, a) => a.length <= 6 || k % Math.ceil(a.length / 5) === 0) : null;
  const f = frame({ xlo: 1, xhi: Math.max(n, 2), ylo, yhi, log, yTicks, xLabel: "acquisition" });
  const xs = c => c.map((_, i) => i + 1);
  const lines = series.map(s => s.curves.map(c => `<path d="${f.path(xs(c), c)}" fill="none" stroke="${colorOf(s.method)}" stroke-dasharray="${dashOf(s.method)}" stroke-width="1" stroke-opacity="0.3"/>`).join("") +
    `<path d="${f.path(xs(s.med), s.med)}" fill="none" stroke="${colorOf(s.method)}" stroke-dasharray="${dashOf(s.method)}" stroke-width="2" stroke-linejoin="round"/>`).join("");
  el("conv").innerHTML = `<svg viewBox="0 0 ${f.W} ${f.H}">${f.axes}${lines}
    <line id="cross" y1="${f.M.t}" y2="${f.H - f.M.b}" stroke="var(--text-muted)" stroke-dasharray="2 2" visibility="hidden"/>
    ${series.map((s, j) => `<circle class="dot" data-j="${j}" r="4" fill="${colorOf(s.method)}" stroke="var(--surface-1)" stroke-width="2" visibility="hidden"/>`).join("")}
    <rect x="${f.M.l}" y="0" width="${f.W - f.M.l - f.M.r}" height="${f.H}" fill="transparent"/></svg>`;
  const svg = el("conv").querySelector("svg"), cross = svg.querySelector("#cross");
  svg.addEventListener("mousemove", e => {
    const box = svg.getBoundingClientRect(), px = (e.clientX - box.left) * f.W / box.width;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - f.M.l) / (f.W - f.M.l - f.M.r) * (Math.max(n, 2) - 1))));
    cross.setAttribute("x1", f.sx(i + 1)); cross.setAttribute("x2", f.sx(i + 1)); cross.setAttribute("visibility", "visible");
    const rows = series.map((s, j) => {
      const dot = svg.querySelector(`.dot[data-j="${j}"]`), v = s.med[i];
      if (v === undefined) { dot.setAttribute("visibility", "hidden"); return ""; }
      dot.setAttribute("cx", f.sx(i + 1)); dot.setAttribute("cy", f.sy(v)); dot.setAttribute("visibility", "visible");
      return `<div class="r">${lineKey(s.method)}${methodLabel(s.method)}<b>${fmt(v)}</b></div>`;
    }).join("");
    tooltip.innerHTML = `<div style="color:var(--text-secondary)">acquisition ${i + 1} · median</div>${rows}`;
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(e.clientX + 14, innerWidth - tooltip.offsetWidth - 8)}px`;
    tooltip.style.top = `${e.clientY + 14}px`;
  });
  svg.addEventListener("mouseleave", () => {
    tooltip.style.display = "none"; cross.setAttribute("visibility", "hidden");
    svg.querySelectorAll(".dot").forEach(d => d.setAttribute("visibility", "hidden"));
  });
}

// ---------- heatmap color scales ----------
const hexRgb = h => { const m = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(m.slice(i, i + 2), 16)); };
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const diverging = () => {
  const lo = hexRgb(css("--series-1")), mid = hexRgb(css("--neutral-mid")), hi = hexRgb(css("--series-2"));
  return v => { const t = Math.max(-1, Math.min(1, v)); return t < 0 ? mix(mid, lo, -t) : mix(mid, hi, t); };
};
const sequential = (lo, hi) => {
  const a = hexRgb(css("--series-1")), b = hexRgb(css("--neutral-mid"));
  return v => mix(a, b, Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1))));
};
function paint(canvas, n, values, scale) {
  canvas.width = n; canvas.height = n;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(n, n);
  // values are ij-indexed (first axis = horizontal); canvas rows run top to bottom
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const [r, g, b] = scale(values[i * n + j]), p = ((n - 1 - j) * n + i) * 4;
    img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// ---------- detail views ----------
function renderDetail(reset = false) {
  stop();
  const run = currentRun();
  if (!run || !run.incumbents.length) {
    el("detail").innerHTML = `<div class="empty">no candidates for this run yet</div>`;
    el("slider").max = 0; el("pos").textContent = ""; el("detail-meta").textContent = "";
    return;
  }
  el("slider").max = run.incumbents.length - 1;
  if (reset || state.k > run.incumbents.length - 1) state.k = run.incumbents.length - 1;
  el("slider").value = state.k;
  drawDetail(run);
}
function drawDetail(run) {
  const info = TARGETS[run.target], k = state.k, idx = run.incumbents[k], value = run.y[idx];
  const shown = run.optimum != null ? `regret ${fmt(value - run.optimum)}` : `return ${fmt(-value)}`;
  el("pos").textContent = `${k + 1}/${run.incumbents.length} improvements`;
  el("detail-meta").textContent = `acquisition ${idx + 1} of ${run.y.length} · ${shown}`;
  const ridge = info.group === "ridge";
  const draw = {
    curve: ridge ? ridgeCurve : projectionCurve, heat: ridge ? ridgeHeat : projectionHeat,
    scatter: projectionScatter, points: ridgePoints, actions: actionsView, video: videoView,
  }[info.view.kind];
  draw(run, info.view, k, idx, colorOf(run.method));
  loadVideos();
}

function projectionCurve(run, view, k, idx, color) {
  const n = view.target.length, xs = view.target.map((_, i) => 2 * (i + 0.5) / n - 1);
  const f = frame({ xlo: -1, xhi: 1, ylo: -1.1, yhi: 1.1, xLabel: "x" });
  const prev = k > 0 ? `<path d="${f.path(xs, run.detail.f[k - 1])}" fill="none" stroke="${color}" stroke-width="1" stroke-opacity="0.35"/>` : "";
  el("detail").innerHTML = `<svg viewBox="0 0 ${f.W} ${f.H}">${f.axes}
    <path d="${f.path(xs, view.target)}" fill="none" stroke="var(--reference)" stroke-width="2" stroke-dasharray="5 3"/>
    ${prev}<path d="${f.path(xs, run.detail.f[k])}" fill="none" stroke="${color}" stroke-width="2"/></svg>
    <div class="legend" style="margin-top:6px"><span><i class="swatch" style="background:var(--reference)"></i>target</span>
    <span><i class="swatch" style="background:${color}"></i>tanh f, best so far</span>
    <span><i class="swatch" style="background:${color};opacity:.35"></i>previous best</span></div>`;
}
function projectionHeat(run, view, k) {
  const n = view.n, fk = run.detail.f[k], residual = fk.map((v, i) => v - view.target[i]);
  el("detail").innerHTML = `<div class="heats">
    <figure><figcaption>target</figcaption><canvas id="h0"></canvas></figure>
    <figure><figcaption>tanh f, best so far</figcaption><canvas id="h1"></canvas></figure>
    <figure><figcaption>residual</figcaption><canvas id="h2"></canvas></figure></div>
    <div class="colorbar">−1<i style="background:linear-gradient(90deg,var(--series-1),var(--neutral-mid),var(--series-2))"></i>+1</div>`;
  const scale = diverging();
  paint(el("h0"), n, view.target, scale); paint(el("h1"), n, fk, scale); paint(el("h2"), n, residual, scale);
}
function projectionScatter(run, view, k, idx, color) {
  const f = frame({ H: 300, xlo: -1, xhi: 1, ylo: -1.05, yhi: 1.05, xLabel: "target at quadrature points" });
  const dots = view.target.map((t, i) => `<circle cx="${f.sx(t).toFixed(1)}" cy="${f.sy(run.detail.f[k][i]).toFixed(1)}" r="2.2" fill="${color}" fill-opacity="0.55"/>`).join("");
  el("detail").innerHTML = `<svg viewBox="0 0 ${f.W} ${f.H}">${f.axes}
    <line x1="${f.sx(-1)}" y1="${f.sy(-1)}" x2="${f.sx(1)}" y2="${f.sy(1)}" stroke="var(--reference)" stroke-dasharray="5 3"/>${dots}</svg>
    <p class="meta" style="margin-top:6px">tanh f (vertical) against the target at 200 quadrature points (d=${TARGETS[run.target].d}); a perfect fit lies on the dashed diagonal.</p>`;
}
function ridgeCurve(run, view, k, idx, color) {
  const n = view.target.length, xs = view.target.map((_, i) => (i + 0.5) / n);
  const lo = Math.min(...view.target), hi = Math.max(...view.target), pad = (hi - lo || 1) * 0.06;
  const f = frame({ xlo: 0, xhi: 1, ylo: lo - pad, yhi: hi + pad, xLabel: "ridge coordinate sigmoid(g)" });
  const coords = run.detail.coords.slice(0, idx + 1);
  const dots = coords.map((c, i) => `<circle cx="${f.sx(c[0]).toFixed(1)}" cy="${f.sy(run.y[i]).toFixed(1)}" r="3" fill="${color}" fill-opacity="0.35"/>`).join("");
  el("detail").innerHTML = `<svg viewBox="0 0 ${f.W} ${f.H}">${f.axes}
    <path d="${f.path(xs, view.target)}" fill="none" stroke="var(--reference)" stroke-width="2"/>${dots}
    <circle cx="${f.sx(coords[idx][0])}" cy="${f.sy(run.y[idx])}" r="6" fill="${color}" stroke="var(--surface-1)" stroke-width="2"/></svg>
    <div class="legend" style="margin-top:6px"><span><i class="swatch" style="background:var(--reference)"></i>profile</span>
    <span><i class="swatch" style="background:${color};opacity:.4"></i>candidates so far</span>
    <span><i class="swatch" style="background:${color}"></i>best so far</span></div>`;
}
function ridgeHeat(run, view, k, idx, color) {
  const n = view.n, lo = Math.min(...view.target), hi = Math.max(...view.target), S = 300;
  const coords = run.detail.coords.slice(0, idx + 1), path = run.incumbents.slice(0, k + 1).map(i => run.detail.coords[i]);
  const p = c => [(c[0] * S).toFixed(1), ((1 - c[1]) * S).toFixed(1)];
  el("detail").innerHTML = `<div style="position:relative;max-width:360px">
    <canvas id="h0"></canvas>
    <svg viewBox="0 0 ${S} ${S}" style="position:absolute;inset:0">
      ${coords.map(c => `<circle cx="${p(c)[0]}" cy="${p(c)[1]}" r="3" fill="${color}" fill-opacity="0.5" stroke="var(--surface-1)" stroke-width="0.8"/>`).join("")}
      <path d="${path.map((c, i) => `${i ? "L" : "M"}${p(c).join(",")}`).join("")}" fill="none" stroke="var(--text-primary)" stroke-width="1.5"/>
      <circle cx="${p(path.at(-1))[0]}" cy="${p(path.at(-1))[1]}" r="6" fill="${color}" stroke="var(--surface-1)" stroke-width="2"/>
    </svg></div>
    <div class="colorbar">${fmt(lo)}<i style="background:linear-gradient(90deg,var(--series-1),var(--neutral-mid))"></i>${fmt(hi)}</div>
    <p class="meta" style="margin-top:6px">profile over the two ridge coordinates (bluer = lower); dots are candidates so far, the line traces the best-so-far path.</p>`;
  paint(el("h0"), n, view.target, sequential(lo, hi));
}
function ridgePoints(run, view, k, idx, color) {
  const coords = run.detail.coords.slice(0, idx + 1), path = run.incumbents.slice(0, k + 1).map(i => run.detail.coords[i]);
  const f = frame({ W: 360, H: 300, xlo: 0, xhi: 1, ylo: 0, yhi: 1, xLabel: "ridge coordinate 1 (vertical: 2)" });
  el("detail").innerHTML = `<svg viewBox="0 0 ${f.W} ${f.H}">${f.axes}
    ${coords.map(c => `<circle cx="${f.sx(c[0]).toFixed(1)}" cy="${f.sy(c[1]).toFixed(1)}" r="3" fill="${color}" fill-opacity="0.4"/>`).join("")}
    <path d="${path.map((c, i) => `${i ? "L" : "M"}${f.sx(c[0]).toFixed(1)},${f.sy(c[1]).toFixed(1)}`).join("")}" fill="none" stroke="var(--text-primary)" stroke-width="1.5"/>
    <circle cx="${f.sx(path.at(-1)[0])}" cy="${f.sy(path.at(-1)[1])}" r="6" fill="${color}" stroke="var(--surface-1)" stroke-width="2"/></svg>
    <p class="meta" style="margin-top:6px">first two of ${TARGETS[run.target].d} ridge coordinates; best so far at (${path.at(-1).map(v => v.toFixed(2)).join(", ")})</p>`;
}
function actionsView(run, view, k, idx, color) {
  const f = frame({ H: 200, xlo: 0, xhi: view.horizon - 1, ylo: -1.1, yhi: 1.1, xLabel: "time step" });
  const u = run.detail.u[k], prev = k > 0 ? run.detail.u[k - 1] : null;
  // one line per action channel, channels told apart by opacity
  const lines = u.map((channel, c) => {
    const op = u.length > 1 ? 1 - 0.6 * c / (u.length - 1) : 1;
    return (prev ? `<path d="${f.path(view.t, prev[c])}" fill="none" stroke="${color}" stroke-width="1" stroke-opacity="${0.25 * op}"/>` : "") +
      `<path d="${f.path(view.t, channel)}" fill="none" stroke="${color}" stroke-width="${u.length > 4 ? 1.2 : 2}" stroke-opacity="${op}"/>`;
  }).join("");
  el("detail").innerHTML = `<svg viewBox="0 0 ${f.W} ${f.H}">${f.axes}${lines}</svg>
    <p class="meta" style="margin:4px 0 8px">open-loop plan tanh u(t), ${u.length} action channel${u.length > 1 ? "s" : ""} (shaded by index), faint = previous best</p>` + videoHtml(run, k);
}
function videoView(run, view, k) { el("detail").innerHTML = videoHtml(run, k); }
function videoHtml(run, k) {
  const best = Object.keys(run.videos || {}).map(Number).sort((a, b) => a - b).at(-1);
  if (best === undefined) return `<div class="empty">no rollout video rendered</div>`;
  return `<video data-src="${run.videos[best]}" autoplay muted loop playsinline controls onerror="videoFailed(this)"></video>
    <p class="meta" style="margin-top:6px">rollout of the best candidate (acquisition ${best + 1})</p>`;
}
// VS Code's HTML preview writes this page into a frame whose media elements cannot reach local files,
// so there the outer frame fetches each video and the page plays it from a cached blob url,
// from next to the page's own webview address (its host differs between local and remote windows)
const inPreview = !/^(file|https?):$/.test(location.protocol) && DATA.root;
const previewSettings = inPreview && window.parent.document.getElementById("preview-html-settings");
const previewBase = previewSettings ? JSON.parse(previewSettings.dataset.settings).src
  : `https://file+.vscode-resource.vscode-cdn.net${encodeURI(DATA.root)}/`;
const videoBlobs = new Map();
function loadVideos() {
  for (const video of el("detail").querySelectorAll("video[data-src]")) {
    const rel = video.dataset.src;
    if (!inPreview) { video.src = rel; continue; }
    if (!videoBlobs.has(rel)) {
      videoBlobs.set(rel, window.parent.fetch(new URL(rel, previewBase))
        .then(r => r.ok ? r.blob() : Promise.reject(new Error(`HTTP ${r.status}`)))
        .then(blob => URL.createObjectURL(new Blob([blob], { type: "video/mp4" }))));
    }
    videoBlobs.get(rel).then(src => video.src = src, error => videoFailed(video, error));
  }
}
// show why a video failed rather than an empty black box
function videoFailed(video, error) {
  const div = document.createElement("div");
  div.className = "empty";
  div.textContent = `video failed: ${error || video.error?.message || "unknown error"} (${video.dataset.src})`;
  video.replaceWith(div);
}

// ---------- player ----------
el("slider").addEventListener("input", e => { stop(); state.k = +e.target.value; const run = currentRun(); if (run) drawDetail(run); });
function stop() { clearInterval(state.playing); state.playing = null; el("play").textContent = "▶"; }
el("play").addEventListener("click", () => {
  if (state.playing) return stop();
  const run = currentRun();
  if (!run || !run.incumbents.length) return;
  if (state.k >= run.incumbents.length - 1) state.k = 0;
  el("play").textContent = "❚❚";
  el("slider").value = state.k; drawDetail(run);
  state.playing = setInterval(() => {
    if (state.k >= run.incumbents.length - 1) return stop();
    state.k++; el("slider").value = state.k; drawDetail(run);
  }, TARGETS[run.target].view.kind === "video" ? 2500 : 700);
});

// ---------- summary table ----------
function renderTable() {
  const head = `<tr><th>Problem</th><th>Method</th><th class="num">Seeds</th><th class="num">Acquisitions</th><th class="num">Median best</th></tr>`;
  const rows = targetsIn(state.tab).flatMap(t => shownMethods().map(m => {
    const runs = runsOf(t).filter(r => r.method === m);
    if (!runs.length) return "";
    const lens = runs.map(r => r.y.length), log = runs[0].optimum != null;
    const finals = runs.map(r => { const b = bestSoFar(r.y).at(-1); return log ? b - r.optimum : -b; });
    const range = Math.min(...lens) === Math.max(...lens) ? lens[0] : `${Math.min(...lens)}–${Math.max(...lens)}`;
    return `<tr><td>${problemLabel(t)}</td><td><i class="swatch" style="background:${colorOf(m)};margin-right:6px;vertical-align:middle"></i>${methodLabel(m)}</td>
      <td class="num">${runs.length}</td><td class="num">${range}</td><td class="num">${fmt(median(finals))} ${log ? "regret" : "return"}</td></tr>`;
  }));
  el("table").innerHTML = head + (rows.join("") || `<tr><td colspan="5" class="empty">No runs yet.</td></tr>`);
}

// repaint canvases when the color scheme flips
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => renderDetail());

// start from #tab/target when given, else the first non-empty tab
const [hashTab, hashTarget] = location.hash.slice(1).split("/");
const firstTab = targetsIn(hashTab || "").length ? hashTab : TABS.map(t => t[0]).find(t => targetsIn(t).length);
if (firstTab) { selectTab(firstTab); if (targetsIn(firstTab).includes(hashTarget)) selectTarget(hashTarget); }
else { renderTabs(); el("detail").innerHTML = el("conv").innerHTML = `<div class="empty">No runs yet.</div>`; }
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/tbd/sweep")
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

    data = build(args.results_dir, args.videos, args.jobs)
    out = args.out or os.path.join(args.results_dir, "dashboard.html")
    with open(out, "w") as f:
        f.write(PAGE.replace("/*DATA*/{}", json.dumps(data)))
    print(f"{len(data['runs'])} runs -> {out}")
