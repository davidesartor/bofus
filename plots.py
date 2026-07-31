import functools
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import pickle

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/neurips")
PLOTS_DIR = os.environ.get("PLOTS_DIR", "plots")

# every running best panel uses this rect, so the tick label gutter is identical
AXES_RECT = (0.22, 0.13, 0.75, 0.79)

# these two are read as a physical quantity, not as a regret, so they stay linear
LINEAR_TARGETS = ("pendulum", "brachistochrone")

# what the minimised y actually is on each problem, in its own units
YLABELS = {
    "brachistochrone": "Travel time (s)",
    "hopper": "Negative apex height (m)",
    "pendulum": "Negative discounted return",
    "pinwheel": r"Angle error $2(1 - \cos \Delta\theta)$",
    "mnist": "Test error",
}
DEFAULT_YLABEL = "Objective value"


def filter_and_rename_methods(
    df: pd.DataFrame, methods: dict[str, str]
) -> pd.DataFrame:
    # filters method column by dict and renames according to dict values
    # one pass to drop the methods we never plot, then one group per kept method
    kept = df[df["method"].isin(methods)]
    groups = dict(list(kept.groupby("method", sort=False)))
    dfs = [kept.iloc[:0]]
    for method, new_method in methods.items():
        # a method that was never run drops out of the figure without this saying so
        if method not in groups:
            print(f"Warning: no rows for method {method}")
            continue
        dfs.append(groups[method].assign(method=new_method))
    return pd.concat(dfs, ignore_index=True)


def median_and_ci(data: np.ndarray, axis: int = 0):
    median = np.median(data, axis=axis)
    qu = np.clip(0.5 + 1.96 * np.sqrt(0.25 / data.shape[axis]), 0, 1)
    ql = np.clip(0.5 - 1.96 * np.sqrt(0.25 / data.shape[axis]), 0, 1)
    ci_upper = np.quantile(data, qu, axis=axis)
    ci_lower = np.quantile(data, ql, axis=axis)
    return median, ci_lower, ci_upper


def runs_by_acquisition(df_method: pd.DataFrame) -> np.ndarray:
    """(runs, acquisitions) grid, taking the runs of each acquisition in row order."""
    i = df_method["i"].to_numpy()
    y = df_method["y"].to_numpy()
    # a stable sort by i keeps the run order within each acquisition, so runs line up
    acquisitions = df_method["i"].nunique()
    return y[np.argsort(i, kind="stable")].reshape(acquisitions, -1).T


def plot_ys(ax, ys: np.ndarray, style: dict):
    runs, max_acquisitions = ys.shape
    y_best = np.minimum.accumulate(ys, axis=-1)
    x = np.arange(1, max_acquisitions + 1)
    m, ll, ul = median_and_ci(y_best, axis=0)
    ax.plot(x, m, **style)
    ax.fill_between(x, ll, ul, alpha=0.2, color=ax.lines[-1].get_color())


def setup_log_yaxis(ax):
    """Force consistent 10^x labels on log y-axis, even for sub-decade ranges.

    Places major ticks at every decade boundary within the current ylim and
    formats them as 10^x. Minor ticks are drawn at 2..9 within each decade
    but left unlabeled, which prevents matplotlib from auto-labeling them as
    2x10^x / 3x10^x when the range spans less than one decade.
    """
    ymin, ymax = ax.get_ylim()
    lo = np.floor(np.log10(ymin))
    hi = np.ceil(np.log10(ymax))

    decades = np.arange(lo, hi + 1)
    ax.yaxis.set_major_locator(mticker.FixedLocator(10.0**decades))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))

    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def widen_ylim_to_decade(ax):
    """Grow a sub-decade log range to a full decade, so a 10^x label lands inside."""
    ymin, ymax = ax.get_ylim()
    span = np.log10(ymax / ymin)
    if span < 1.0:
        pad = (1.0 - span) / 2
        ax.set_ylim(10 ** (np.log10(ymin) - pad), 10 ** (np.log10(ymax) + pad))


@functools.cache
def brachistochrone_optimal_time() -> float:
    """Cycloid travel time, the constant the recorded brachistochrone y is offset by."""
    from src.targets import Brachistochrone

    _, optimal_time = Brachistochrone().find_brachistochrone()
    return float(optimal_time)


def plot_running_best(
    df: pd.DataFrame, title: str, save_dir: str, methods: dict[str, str]
):
    os.makedirs(save_dir, exist_ok=True)
    df = filter_and_rename_methods(df, methods)

    for target_fn, df_targ in tqdm(list(df.groupby("target_fn", sort=False))):
        # the run records time above the cycloid, the plot shows the travel time itself
        offset = (
            brachistochrone_optimal_time() if target_fn == "brachistochrone" else 0.0
        )
        fig = plt.figure(figsize=(4, 4))
        # fixed axes rect, not a tight bbox, so every panel reserves the same
        # width for the y tick labels however wide they turn out to be
        ax = fig.add_axes(AXES_RECT)
        for (method, df_method), color in zip(
            df_targ.groupby("method", sort=False), plt.cm.tab10.colors
        ):
            ys = runs_by_acquisition(df_method) + offset  # shape (runs, acquisitions)
            plot_ys(ax, ys, style={"color": color, "label": method})

        if target_fn == "brachistochrone":
            ax.axhline(offset, color="black", linestyle=":", label="cycloid")

        ax.set_title(f"{target_fn}{title}")
        ax.set_xlabel("Acquisitions")
        ax.set_ylabel(YLABELS.get(target_fn, DEFAULT_YLABEL))
        # hopper minimises a negative reward, so a log axis has nothing it can draw there
        log_scale = df_targ["y"].min() > 0 and target_fn not in LINEAR_TARGETS
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlim(1, df_targ["i"].max() + 1)

        if log_scale:
            widen_ylim_to_decade(ax)
            setup_log_yaxis(ax)

        ax.grid()
        ax.legend(loc="upper right")
        # no tight bbox: it would crop each panel down to its own label widths
        plt.savefig(f"{save_dir}/{target_fn}.pdf")
        plt.close()


def print_results_table(
    df: pd.DataFrame,
    save_dir: str,
    methods: dict[str, str],
    metrics: list[str] = ["best_y", "avg_regret"],
):
    df = filter_and_rename_methods(df, methods)

    target_fns = sorted(df["target_fn"].unique())
    method_names = df["method"].unique().tolist()

    def get_cell(method, target_fn, metric):
        vals = df[(df["method"] == method) & (df["target_fn"] == target_fn)][
            metric
        ].values
        if len(vals) == 0:
            return "N/A"
        median, lo, hi = median_and_ci(vals)
        return f"{median:.4g} [{lo:.4g}, {hi:.4g}]"

    for metric in metrics:
        rows = pd.DataFrame(
            {
                target_fn: [get_cell(m, target_fn, metric) for m in method_names]
                for target_fn in target_fns
            },
            index=method_names,
        )
        with open(f"{save_dir}/{metric}_table.txt", "w") as f:
            f.write(rows.to_string())


def filter_best_profile_per_lengthscale(df: pd.DataFrame) -> pd.DataFrame:
    # best profile per (target_fn, method, lengthscale), so no method is handicapped by a shared profile
    best_profile = (
        df.groupby(["target_fn", "method", "lengthscale", "profile"])["best_y"]
        .median()
        .reset_index()
        .loc[
            lambda d: d.groupby(["target_fn", "method", "lengthscale"])[
                "best_y"
            ].idxmin()
        ][["target_fn", "method", "lengthscale", "profile"]]
    )
    return df.merge(best_profile, on=["target_fn", "method", "lengthscale", "profile"])


def markdown_table(rows: list[list[str]], headers: list[list[str]]) -> str:
    """GitHub-flavoured markdown table, the format OpenReview accepts.

    Only the first header row is a real header; any further ones are emitted as
    leading body rows, which is as close to a spanning header as markdown gets.
    """
    header, *subheaders = headers
    alignment = [":--"] + ["--:"] * (len(header) - 1)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(alignment) + " |",
        *["| " + " | ".join(row) + " |" for row in [*subheaders, *rows]],
    ]
    return "\n".join(lines)


METRIC_LABELS = {"best_y": "final y", "avg_regret": "avg regret"}


def print_lengthscale_tables(
    df: pd.DataFrame,
    save_dir: str,
    methods: dict[str, str],
    metrics: list[str] = ["best_y", "avg_regret"],
    drop_target_fns: tuple[str, ...] = (),
):
    """One table per lengthscale, to show the ranking does not depend on the chosen lengthscale."""
    os.makedirs(save_dir, exist_ok=True)
    df = filter_and_rename_methods(df, methods)
    df = df[~df["target_fn"].isin(drop_target_fns)]
    df = filter_best_profile_per_lengthscale(df)

    target_fns = sorted(df["target_fn"].unique())
    method_names = [m for m in methods.values() if (df["method"] == m).any()]
    lengthscales = sorted(df["lengthscale"].unique())
    columns = pd.MultiIndex.from_tuples(
        [(method, metric) for method in method_names for metric in metrics]
    )
    # markdown has no spanning header, so the method name sits above its first metric
    method_header = [
        f"**{method}**" if metric == metrics[0] else "" for method, metric in columns
    ]
    metric_header = [METRIC_LABELS.get(metric, metric) for _, metric in columns]

    def header_rows(corner: str) -> list[list[str]]:
        return [[corner] + method_header, [""] + metric_header]

    def cells(df_scale):
        empty = lambda dtype: pd.DataFrame(
            index=target_fns, columns=columns, dtype=dtype
        )
        medians, lowers, uppers = empty(float), empty(float), empty(float)
        for target_fn in target_fns:
            for method, metric in columns:
                vals = df_scale[
                    (df_scale["target_fn"] == target_fn)
                    & (df_scale["method"] == method)
                ][metric].values
                if len(vals) == 0:
                    continue
                median, lo, hi = median_and_ci(vals)
                medians.loc[target_fn, (method, metric)] = median
                lowers.loc[target_fn, (method, metric)] = lo
                uppers.loc[target_fn, (method, metric)] = hi
        return medians, lowers, uppers

    def tolerant_ranks(medians, lowers, uppers, metric_columns, target_fn):
        """Competition rank where a method ties with anything better whose median its own CI covers."""
        ranks = {}
        for column in metric_columns:
            median = medians.loc[target_fn, column]
            if pd.isna(median):
                continue
            # the tie is judged by the CI of the worse method, so a noisy runner up keeps the rank
            beaten_by = [
                other
                for other in metric_columns
                if other != column
                and pd.notna(medians.loc[target_fn, other])
                and medians.loc[target_fn, other] < median
                and not (
                    lowers.loc[target_fn, column]
                    <= medians.loc[target_fn, other]
                    <= uppers.loc[target_fn, column]
                )
            ]
            ranks[column] = 1 + len(beaten_by)
        return pd.Series(ranks, dtype=float)

    def format_rows(medians: pd.DataFrame):
        """Factor a common power of ten out of each problem, so every cell is 3 significant digits."""
        text = pd.DataFrame("N/A", index=medians.index, columns=columns, dtype=object)
        exponents = {}
        for target_fn, row in medians.iterrows():
            largest = np.abs(row.dropna()).max() if row.notna().any() else 0.0
            exponent = int(np.floor(np.log10(largest))) if largest > 0 else 0
            exponents[target_fn] = exponent
            for column in columns:
                if pd.notna(row[column]):
                    text.loc[target_fn, column] = f"{row[column] / 10**exponent:.2f}"
        return text, exponents

    # both metrics are minimized, so the best method on a problem is the smallest median
    mean_ranks = pd.DataFrame(index=lengthscales, columns=columns, dtype=float)
    result_rows = []
    for lengthscale in lengthscales:
        medians, lowers, uppers = cells(df[df["lengthscale"] == lengthscale])
        text, exponents = format_rows(medians)
        markdown = text.copy()

        # every method tied for rank 1 on a problem is highlighted, not just the smallest median
        all_ranks = pd.DataFrame(index=target_fns, columns=columns, dtype=float)
        for metric in metrics:
            metric_columns = [c for c in columns if c[1] == metric]
            for target_fn in target_fns:
                ranks = tolerant_ranks(
                    medians, lowers, uppers, metric_columns, target_fn
                )
                all_ranks.loc[target_fn, ranks.index] = ranks
                for column in ranks[ranks == 1].index:
                    markdown.loc[target_fn, column] = (
                        "**" + text.loc[target_fn, column] + "**"
                    )
        ranks = all_ranks.mean(axis=0)

        mean_ranks.loc[lengthscale] = ranks
        markdown.loc["*mean rank*"] = [f"*{ranks[c]:.2f}*" for c in columns]

        # one stacked table, so each lengthscale is a section under a shared header
        result_rows.append([f"**rho = {lengthscale:g}**"] + [""] * len(columns))
        for name, row in markdown.iterrows():
            label = f"{name} (x10^{exponents[name]})" if name in exponents else name
            result_rows.append([label] + list(row))

    # summary: mean rank of each method at every lengthscale
    rank_text = mean_ranks.map(lambda v: f"{v:.2f}")
    for metric in metrics:
        metric_columns = [c for c in columns if c[1] == metric]
        for lengthscale, column in (
            mean_ranks[metric_columns].idxmin(axis=1).dropna().items()
        ):
            rank_text.loc[lengthscale, column] = (
                "**" + rank_text.loc[lengthscale, column] + "**"
            )
    rank_rows = [
        [f"**rho = {lengthscale:g}**"] + list(row)
        for lengthscale, row in rank_text.iterrows()
    ]

    report = [
        "## Sensitivity of the ranking to the lengthscale rho",
        "",
        "Mean rank over problems at each lengthscale. Lower is better, best per "
        "lengthscale and metric in bold. A stable ordering down the rows means the "
        "ranking does not depend on the chosen lengthscale.",
        "",
        markdown_table(rank_rows, header_rows("rho")),
        "",
        "### Full results",
        "",
        "Median over seeds, each method using its own best kernel profile at that "
        "lengthscale. Each problem carries a common power of ten in its row label, so "
        "a cell of 1.65 in a row labelled (x10^1) means 16.5. "
        "Lower is better for both metrics. Methods are ranked per problem and metric, "
        "with a method tied with every better one whose median falls inside its own "
        "95% CI; everything tied for first is in bold. Mean rank is over the problems "
        "on which the method ran.",
        "",
        markdown_table(result_rows, header_rows("Problem")),
        "",
    ]
    with open(f"{save_dir}/lengthscale_ablation.md", "w") as f:
        f.write("\n".join(report))


def plot_timing_boxes(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
    """Box plots of per-run surrogate fit and acquisition times, one file per target."""
    os.makedirs(save_dir, exist_ok=True)
    df = filter_and_rename_methods(df, methods)
    metrics = {"t_fit": "Surrogate fit", "t_acq": "Acquisition"}

    for target_fn in tqdm(sorted(df["target_fn"].unique())):
        df_targ = df[df["target_fn"] == target_fn]
        method_names = [m for m in methods.values() if (df_targ["method"] == m).any()]

        fig, axes = plt.subplots(
            1, len(metrics), figsize=(4 * len(metrics), 4), sharey=True
        )
        for ax, (metric, label) in zip(axes, metrics.items()):
            data = [
                df_targ[df_targ["method"] == method][metric].values
                for method in method_names
            ]
            boxes = ax.boxplot(
                data, tick_labels=method_names, showfliers=False, patch_artist=True
            )
            for patch, color in zip(boxes["boxes"], plt.cm.tab10.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            for median in boxes["medians"]:
                median.set_color("black")

            ax.set_title(label)
            ax.set_yscale("log")
            ax.grid(axis="y")
            ax.tick_params(axis="x", rotation=45)

        axes[0].set_ylabel("Time (s)")
        fig.suptitle(target_fn)
        plt.savefig(f"{save_dir}/{target_fn}.pdf", bbox_inches="tight")
        plt.close()


def plot_mnist_activations(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
    os.makedirs(save_dir, exist_ok=True)
    df = df[df["target_fn"] == "mnist"]
    method_colors = {m: f"C{i}" for i, m in enumerate(methods.keys())}

    x = jnp.linspace(-3, 3, 1000)
    plt.figure(figsize=(8, 4))

    for method, label in tqdm(methods.items()):
        color = method_colors[method]
        df_method = df[df["method"] == method][
            ["profile", "lengthscale", "seed", "best_y"]
        ].sort_values("best_y")
        ys = []
        for _, row in df_method.iterrows():
            path = f"{RESULTS_DIR}/mnist/{method}/{row['profile']}_lengthscale_{row['lengthscale']}/seed_{row['seed']}.pkl"
            res = pickle.load(open(path, "rb"))
            f = res["observation_locations"][res["observation_values"].argmin()]
            activation = jax.vmap(lambda x: f((x[None] + 1.0) / 2.0) + jax.nn.relu(x))
            y = activation(x)
            ys.append(y)
            plt.plot(x, y, color=color, alpha=0.2)

        if ys:
            plt.plot(
                x, jnp.stack(ys).mean(axis=0), color=color, linewidth=3, label=label
            )

    plt.plot(x, jax.nn.relu(x), "k:", label="ReLU", linewidth=2)
    plt.xlim(-3, 3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.savefig(f"{save_dir}/mnist_activations.pdf", bbox_inches="tight")
    plt.close()


def plot_brachistochrone_path(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
    os.makedirs(save_dir, exist_ok=True)
    df = df[df["target_fn"] == "brachistochrone"]
    from src.targets import Brachistochrone

    targ = Brachistochrone()
    method_colors = {m: f"C{i}" for i, m in enumerate(methods.keys())}

    x0, y0 = targ.initial_position
    x1, y1 = targ.final_position
    x = jnp.linspace(x0, x1, 1000)

    plt.figure(figsize=(8, 4))
    cycloid, optimal_time = targ.find_brachistochrone()

    for method, label in tqdm(methods.items()):
        color = method_colors[method]
        df_method = df[df["method"] == method][
            ["profile", "lengthscale", "seed", "best_y"]
        ].sort_values("best_y")
        ys = []
        for _, row in df_method.iterrows():
            path = f"{RESULTS_DIR}/brachistochrone/{method}/{row['profile']}_lengthscale_{row['lengthscale']}/seed_{row['seed']}.pkl"
            res = pickle.load(open(path, "rb"))
            f = res["observation_locations"][res["observation_values"].argmin()]
            curve = jax.vmap(targ.get_curve(f))
            y = curve(x)
            ys.append(y)
            plt.plot(x, y, color=color, alpha=0.2)

        if ys:
            plt.plot(
                x, jnp.stack(ys).mean(axis=0), color=color, linewidth=3, label=label
            )

    plt.plot(x, cycloid(x), "k:", label=f"Cycloid", linewidth=2)
    plt.plot([x0, x1], [y0, y1], "ro")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.savefig(f"{save_dir}/brachistochrone_path.pdf", bbox_inches="tight")
    plt.close()


def animate_hopper(df: pd.DataFrame, save_dir: str, methods: dict[str, str]):
    """Animate the best hopper run of each method, as a gif."""
    os.makedirs(save_dir, exist_ok=True)
    df = df[df["target_fn"] == "hopper"]
    from matplotlib.animation import PillowWriter
    from src.targets.hopper import HoppingRobot, keypoints

    robot = HoppingRobot()
    ts = np.arange(robot.n_steps) / robot.n_steps

    for method, label in tqdm(methods.items()):
        df_method = df[df["method"] == method][
            ["profile", "lengthscale", "seed", "best_y"]
        ].sort_values("best_y")
        if df_method.empty:
            print(f"Warning: no rows for method {method}")
            continue

        row = df_method.iloc[0]
        path = f"{RESULTS_DIR}/hopper/{method}/{row['profile']}_lengthscale_{row['lengthscale']}/seed_{row['seed']}.pkl"
        res = pickle.load(open(path, "rb"))
        f = res["observation_locations"][res["observation_values"].argmin()]

        qs, _, _ = robot.rollout(f)
        points = np.asarray(jax.vmap(keypoints, in_axes=(0, None))(qs, robot.params))
        reference = np.asarray(robot.amplitude * jax.vmap(f)(jnp.array(ts)[:, None]))
        apex_height = -float(robot(f))

        fig, (ax, ax_ref) = plt.subplots(
            1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]}
        )
        ax.set_xlim(points[:, :, 0].min() - 0.4, points[:, :, 0].max() + 0.4)
        ax.set_ylim(-0.2, points[:, :, 1].max() + 0.3)
        ax.set_aspect("equal")
        ax.axhline(0.0, color="k", linewidth=2)
        ax.axhline(apex_height, color="C2", linestyle=":", linewidth=1.5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{label}: apex height = {apex_height:.3f} m")
        (leg,) = ax.plot([], [], "-o", color="C0", linewidth=3, markersize=5)
        (foot,) = ax.plot([], [], "-", color="C0", linewidth=4)
        (trace,) = ax.plot([], [], color="C1", alpha=0.6, linewidth=1)
        clock = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        ax_ref.plot(ts, reference[:, 0], color="C3", label="hip")
        ax_ref.plot(ts, reference[:, 1], color="C4", label="knee")
        ax_ref.set_xlabel("normalized time")
        ax_ref.set_ylabel("reference offset [rad]")
        ax_ref.grid()
        ax_ref.legend()
        cursor = ax_ref.axvline(0.0, color="k", linewidth=1)

        writer = PillowWriter(fps=25)
        with writer.saving(fig, f"{save_dir}/hopper_{method}.gif", dpi=100):
            for t in range(0, len(points), max(len(points) // 200, 1)):
                base, hip, knee, ankle, toe, heel = points[t]
                leg.set_data(*np.stack([base, hip, knee, ankle], axis=-1))
                foot.set_data(*np.stack([toe, heel], axis=-1))
                trace.set_data(points[: t + 1, 0, 0], points[: t + 1, 0, 1])
                cursor.set_xdata([ts[t], ts[t]])
                clock.set_text(f"t = {t * robot.dt:.2f} s")
                writer.grab_frame()
        plt.close(fig)


def make_ridge(name: str):
    """Ridge target, matching the definitions in run.py."""
    import vlse

    from src import targets

    profiles = {
        "gramacylee": vlse.GramacyLee(normalized=True),
        "ackley": vlse.Ackley(d=2, normalized=True),
        "hartmann": vlse.Hartmann3(normalized=True),
        "rosenbrock": vlse.Rosenbrock(d=4, normalized=True),
        "michalewicz": vlse.Michalewicz(d=5, normalized=True),
    }
    return targets.Ridge(profiles[name])


def sample_rkhs_functions(
    d: int, n: int, basis_points: int, lengthscale: float, seed: int
) -> list:
    """Random RKHS functions, drawn like the initial acquisitions in run.py."""
    from src import kernels, rkhs

    kernel = rkhs.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=jnp.array([lengthscale] * d),
    )
    sampler = sp.stats.qmc.LatinHypercube(
        d=basis_points * (kernel.d + kernel.m), rng=np.random.default_rng(seed)
    )
    return [
        rkhs.Function.from_array(kernel, p.reshape(basis_points, kernel.d + kernel.m))
        for p in sampler.random(n=n)
    ]


def monte_carlo_evaluator(target, kernel):
    """Target on a fresh n-point grid, bypassing the closed form Ridge would take.

    Grid, weights and quadrature sit in one jit keyed on n, so it compiles once per n
    rather than paying a dispatch per example, and the (d, modes, n) array weights_at
    broadcasts to is never materialized. f travels as its arrays, the kernel is shared.
    """
    from src import rkhs, targets

    @functools.partial(jax.jit, static_argnames="n")
    def value(n: int, key, x: jnp.ndarray, a: jnp.ndarray):
        points = jr.uniform(key, (n, target.d), minval=0.0, maxval=1.0)
        f = rkhs.Function(kernel=kernel, x=x, a=a)
        args = (target.profile, points, target.weights_at(points), target.b)
        return targets.quadrature_value(*args, f)

    def monte_carlo_value(n: int, seed: int, f) -> float:
        return float(value(n, jr.key(seed), f.x, f.a))

    return monte_carlo_value


def convergence_curves(
    name: str,
    n_points: list[int],
    n_examples: int,
    basis_points: int,
    lengthscale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo values on each n, and the closed form value, for every example."""
    target = make_ridge(name)
    fs = sample_rkhs_functions(target.d, n_examples, basis_points, lengthscale, seed=0)
    monte_carlo_value = monte_carlo_evaluator(target, fs[0].kernel)

    y_exact = np.array([float(target.exact_value(f)) for f in fs])
    ys = np.stack(
        [
            [monte_carlo_value(n, seed=i, f=f) for i, f in enumerate(fs)]
            for n in tqdm(n_points, desc=name, leave=False)
        ],
        axis=-1,
    )  # shape (examples, n_points)
    return ys, y_exact


def plot_ridge_convergence_panel(
    name: str,
    save_dir: str,
    n_points: list[int],
    n_examples: int,
    basis_points: int,
    lengthscale: float,
):
    """One problem's convergence panel, self contained so it can run in its own process."""
    ys, y_exact = convergence_curves(
        name, n_points, n_examples, basis_points, lengthscale
    )
    np.savez(f"{save_dir}/{name}.npz", n=n_points, ys=ys, y_exact=y_exact)

    # normalize each example by its own reference value, so they share one axis
    median, lower, upper = median_and_ci(ys / y_exact[:, None], axis=0)

    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.axhline(1.0, color="black", linestyle=":", label="closed form")
    ax.plot(n_points, median, color="tab:blue", label="Monte Carlo")
    ax.fill_between(n_points, lower, upper, alpha=0.2, color="tab:blue")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Evaluation points $n$")
    ax.set_ylabel("target value / closed form value")
    ax.set_title(name)
    ax.grid()
    ax.legend()
    plt.savefig(f"{save_dir}/{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_ridge_convergence(
    save_dir: str,
    n_points: list[int],
    n_examples: int = 32,
    basis_points: int = 10,
    lengthscale: float = 0.2,
):
    """Convergence of the n-point Monte Carlo ridge targets to their exact integral."""
    os.makedirs(save_dir, exist_ok=True)
    names = ["gramacylee", "ackley", "hartmann", "rosenbrock", "michalewicz"]

    # the quadratures barely use one core each, and the problems share nothing
    panel = functools.partial(
        plot_ridge_convergence_panel,
        save_dir=save_dir,
        n_points=n_points,
        n_examples=n_examples,
        basis_points=basis_points,
        lengthscale=lengthscale,
    )
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as pool:
        list(tqdm(pool.map(panel, names), total=len(names)))


if __name__ == "__main__":
    ##############################################################################
    # load data
    print("Loading data...")

    def load(name: str) -> pd.DataFrame:
        """The labels are stored as the categorical directory names: as categoricals every
        groupby below would widen back to the categories of the whole sweep, and the
        lengthscale is a number everywhere here."""
        df = pd.read_parquet(f"{RESULTS_DIR}/{name}.parquet")
        df = df.astype({c: "object" for c in df.select_dtypes("category")})
        return df.assign(lengthscale=pd.to_numeric(df["lengthscale"]))

    summary_all = load("summary_all")
    summary_filtered = load("summary_filtered")
    ys_all = load("ys_all")
    ys_filtered = load("ys_filtered")

    # for each fn, get the best lengthscale/profile combination for each method
    # in summary filtered this is the unique combination left in the df
    print("Printing tables...")
    os.makedirs(f"{PLOTS_DIR}/tables", exist_ok=True)
    best_combos = (
        summary_filtered.groupby(["target_fn", "method"])[["lengthscale", "profile"]]
        .first()
        .reset_index()
    )
    best_combos = filter_and_rename_methods(
        best_combos,
        methods={
            "ours_no_natural_grad": "Ours",
            "kundu": "Kundu",
            "vien": "Vien",
            "shilton": "Shilton",
        },
    )
    with open(f"{PLOTS_DIR}/tables/best_lengthscales.txt", "w") as f:
        for target_fn in best_combos["target_fn"].unique():
            f.write(f"{target_fn}:\n")
            for method in best_combos[
                best_combos["target_fn"] == target_fn
            ].itertuples():
                f.write(
                    f"  {method.method}: lengthscale={method.lengthscale}, profile={method.profile}\n"
                )

    # get only the best lengthscale for each target_fn
    best_lengthscales = summary_filtered.groupby(["target_fn"])["lengthscale"].first()

    ################################################################################
    # METHOD COMPARISON
    print("Plotting method comparison...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"{PLOTS_DIR}/method_comparison",
        methods={
            "ours_no_natural_grad": "ours",
            "kundu": "Kundu",
            "vien": "Vien",
            "shilton": "Shilton",
            "vellanky": "Vellanki",
        },
    )

    ################################################################################
    # NATURAL GRADIENT ABLATIONS
    print("Plotting natural gradient ablations...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"{PLOTS_DIR}/natural_gradient_ablation_ours",
        methods={
            f"ours": f"Natural Gradient",
            f"ours_no_natural_grad": f"Euclidean Gradient",
        },
    )
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"{PLOTS_DIR}/natural_gradient_ablation_vien",
        methods={
            f"vien": f"Natural Gradient",
            f"vien_no_natural_grad": f"Euclidean Gradient",
        },
    )

    ################################################################################
    # PRECONDITIONER REGULARIZATION ABLATION
    # truncation drops the near-null directions outright, so plotting it against both the plain
    # pseudoinverse and the Euclidean gradient separates "the solve is unstable" from "the metric
    # itself is the wrong one"
    print("Plotting preconditioner regularization ablation...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"{PLOTS_DIR}/preconditioner_regularization_ablation",
        methods={
            f"ours_no_natural_grad": f"Euclidean gradient",
            f"ours": f"pseudoinverse",
            f"ours_rcond_1e-10": f"truncated at $10^{{-10}} s_1$",
            f"ours_rcond_1e-08": f"truncated at $10^{{-8}} s_1$",
            f"ours_rcond_1e-06": f"truncated at $10^{{-6}} s_1$",
        },
    )

    ################################################################################
    # CONSTRAINED SEARCH ABLATIONS
    print("Plotting constrained search ablations...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"{PLOTS_DIR}/constrained_search_ablation",
        methods={
            f"ours_no_natural_grad": f"ours",
            f"ours_no_natural_grad_constrain_order": f"ours (constrained search)",
            f"ours": f"ours (natural gradient)",
            f"ours_constrain_order": f"ours (constrained search + natural gradient)",
        },
    )

    ################################################################################
    # CANDIDATES SAMPLING ABLATIONS
    print("Plotting candidates sampling ablations...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"{PLOTS_DIR}/candidates_sampling_ablation",
        methods={
            f"ours": f"Ours",
            f"ours_sample_from_gp": f"Ours (sample from GP)",
            f"shilton": f"Shilton",
        },
    )

    ################################################################################
    # SHILTON REDUCED GRID ABLATION
    print("Plotting shilton reduced grid ablation...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"{PLOTS_DIR}/shilton_reduced_grid_ablation",
        methods={
            f"ours_no_natural_grad": f"ours",
            f"shilton": f"Shilton (full grid)",
            f"shilton_reduced_grid": f"Shilton (reduced grid)",
        },
    )

    ################################################################################
    # BASIS SIZE SCHEDULE ABLATIONS
    print("Plotting basis size schedule ablations...")
    plot_running_best(
        df=ys_filtered,
        title=f"",
        save_dir=f"{PLOTS_DIR}/fixed_k_ablation",
        methods={
            f"ours": f"Ours (growing $k$)",
            f"ours_fixed_k_1": f"$k=1$",
            f"ours_fixed_k_5": f"$k=5$",
            f"ours_fixed_k_10": f"$k=10$",
        },
    )

    ##############################################################################
    # KERNEL PROFILE ABLATIONS
    print("Plotting kernel profile ablations...")
    # merge first: tagging the profile onto the method is a string join per row
    df = ys_all.merge(best_lengthscales.reset_index(), on=["target_fn", "lengthscale"])
    df = df[df["method"] == "ours_no_natural_grad"]
    df = df.assign(method=df["method"] + "_" + df["profile"])
    plot_running_best(
        df=df,
        title=f"",
        save_dir=f"{PLOTS_DIR}/kernel_profile_ablation",
        methods={
            f"ours_no_natural_grad_{profile}": f"{profile}"
            for profile in ys_all["profile"].unique()
        },
    )

    ###############################################################################
    # TABLES AVGREGRET AND BEST_Y
    print("Printing tables...")
    print_results_table(
        df=summary_filtered,
        save_dir=f"{PLOTS_DIR}/tables",
        methods={
            "random": "Random",
            "ours_no_natural_grad": "Ours",
            "kundu": "Kundu",
            "vien": "Vien",
            "shilton": "Shilton",
            "vellanky": "Vellanki",
        },
    )

    ###############################################################################
    # PER LENGTHSCALE TABLES
    print("Printing per lengthscale tables...")
    print_lengthscale_tables(
        df=summary_all,
        save_dir=f"{PLOTS_DIR}/tables",
        methods={
            "random": "Random",
            "ours_no_natural_grad": "Ours",
            "kundu": "Kundu",
            "vien": "Vien",
            "shilton": "Shilton",
            "vellanky": "Vellanki",
        },
        drop_target_fns=("sinc1d", "sinc2d", "sinc3d", "sinc4d"),
    )

    ###############################################################################
    # TIMING TABLES AND BOX PLOTS
    print("Printing timing tables...")
    timing_methods = {
        "ours_no_natural_grad": "Ours",
        "kundu": "Kundu",
        "vien": "Vien",
        "shilton": "Shilton",
        "vellanky": "Vellanki",
    }
    print_results_table(
        df=summary_filtered,
        save_dir=f"{PLOTS_DIR}/tables",
        methods=timing_methods,
        metrics=["t_fit", "t_acq"],
    )

    print("Plotting timing box plots...")
    plot_timing_boxes(
        df=summary_filtered,
        save_dir=f"{PLOTS_DIR}/timings",
        methods=timing_methods,
    )

    ##############################################################################
    # MNIST LEARNED ACTIVATION
    print("Plotting mnist learned activations...")
    plot_mnist_activations(
        df=summary_filtered,
        save_dir=f"{PLOTS_DIR}/f_visualizations",
        methods={
            "ours_no_natural_grad": "Learned Activation",
            # "kundu": "kundu",
            # "vien": "Vien",
            # "shilton": "Shilton",
            # "vellanky": "Vellanky",
        },
    )

    ##############################################################################
    # HOPPER BEST RUN ANIMATION
    print("Animating hopper best runs...")
    animate_hopper(
        df=summary_filtered,
        save_dir=f"{PLOTS_DIR}/f_visualizations",
        methods={
            "ours_no_natural_grad": "Ours",
            # "kundu": "Kundu",
            # "vien": "Vien",
            # "shilton": "Shilton",
            # "vellanky": "Vellanki",
        },
    )

    ##############################################################################
    # BRACHISTOCHRONE LEARNED PATH
    print("Plotting brachistochrone learned path...")
    plot_brachistochrone_path(
        df=summary_filtered,
        save_dir=f"{PLOTS_DIR}/f_visualizations",
        methods={
            "ours_no_natural_grad": "Learned Path",
            # "kundu": "kundu",
            # "vien": "Vien",
            # "shilton": "Shilton",
            # "vellanky": "Vellanky",
        },
    )

    ##############################################################################
    # RIDGE MONTE CARLO CONVERGENCE
    print("Plotting ridge convergence...")
    plot_ridge_convergence(
        save_dir=f"{PLOTS_DIR}/convergence",
        n_points=[2**i for i in range(17)],  # 1 ... 65536
    )
