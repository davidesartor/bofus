"""Appendix ablation grids of the camera-ready paper, one PDF per ablation with a shared legend."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bofus.plots import (
    LINEAR_TARGETS,
    RESULTS_DIR,
    brachistochrone_optimal_time,
    filter_and_rename_methods,
    plot_ys,
    runs_by_acquisition,
    setup_log_yaxis,
    widen_ylim_to_decade,
)

YS = f"{RESULTS_DIR}/ys_filtered.parquet"
YS_ALL = f"{RESULTS_DIR}/ys_all.parquet"
SUMMARY = f"{RESULTS_DIR}/summary_filtered.parquet"
FIGURES = os.environ.get("FIGURES_DIR", "latex/neurips/figures")
INITIAL_ACQUISITIONS = 10

RIDGES = ["gramacylee", "ackley", "hartmann", "rosenbrock", "michalewicz"]
REALWORLD = ["brachistochrone", "mnist", "pendulum", "pinwheel", "hopper"]
SINC = ["sinc1d", "sinc2d", "sinc3d", "sinc4d", "sinc5d"]
TITLES = {
    "gramacylee": "GramacyLee ($d{=}1$)",
    "ackley": "Ackley ($d{=}2$)",
    "hartmann": "Hartmann ($d{=}3$)",
    "rosenbrock": "Rosenbrock ($d{=}4$)",
    "michalewicz": "Michalewicz ($d{=}5$)",
    "brachistochrone": "Brachistochrone (s)",
    "mnist": "MNIST (test error)",
    "pendulum": "Pendulum (neg. return)",
    "pinwheel": "Pinwheel (angle error)",
    "hopper": "Hopper (neg. apex, m)",
    "sinc1d": "Sinc-1D",
    "sinc2d": "Sinc-2D",
    "sinc3d": "Sinc-3D",
    "sinc4d": "Sinc-4D",
    "sinc5d": "Sinc-5D",
}
ABLATIONS = {
    "natural_gradient_ablation_ours": {
        "ours_no_natural_grad": "Euclidean gradient",
        "ours": "Natural gradient",
    },
    "natural_gradient_ablation_vien": {
        "vien_no_natural_grad": "Euclidean gradient",
        "vien": "Natural gradient",
    },
    "preconditioner_regularization_ablation": {
        "ours_no_natural_grad": "Euclidean gradient",
        "ours": "Pseudoinverse",
        "ours_rcond_1e-10": "Truncated at $10^{-10}s_1$",
        "ours_rcond_1e-08": "Truncated at $10^{-8}s_1$",
        "ours_rcond_1e-06": "Truncated at $10^{-6}s_1$",
    },
    "candidates_sampling_ablation": {
        "ours": "L0MO (LHS starts)",
        "ours_sample_from_gp": "L0MO (GP prior starts)",
        "shilton": "Shilton",
    },
    "fixed_k_ablation": {
        "ours": "Growing $k$",
        "ours_fixed_k_1": "Fixed $k=1$",
        "ours_fixed_k_5": "Fixed $k=5$",
        "ours_fixed_k_10": "Fixed $k=10$",
    },
}
PROFILES = {
    "rbf": "RBF",
    "matern52": "Matérn-5/2",
    "matern32": "Matérn-3/2",
    "matern12": "Matérn-1/2",
}

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
    }
)


def panel(ax, df_targ: pd.DataFrame, target: str, colors: dict[str, tuple]):
    """Median running best with 95% CI of every method on one target."""
    offset = brachistochrone_optimal_time() if target == "brachistochrone" else 0.0
    for method, df_method in df_targ.groupby("method", sort=False):
        ys = runs_by_acquisition(df_method) + offset
        plot_ys(ax, ys, style={"color": colors[method], "label": method, "lw": 1.0})

    ax.set_title(TITLES[target], pad=3)
    ax.set_xlim(1, df_targ["i"].max() + 1)
    ax.set_xticks([1, 50, 100] if df_targ["i"].max() > 60 else [1, 20, 40])
    log_scale = df_targ["y"].min() > 0 and target not in LINEAR_TARGETS + (
        "mnist",
        "michalewicz",
    )
    if log_scale:
        ax.set_yscale("log")
        widen_ylim_to_decade(ax)
        setup_log_yaxis(ax)

    # on the real-world tasks the initial designs dwarf the differences that matter
    if target in REALWORLD:
        bottom = ax.dataLim.y0
        top = max(
            line.get_ydata()[INITIAL_ACQUISITIONS - 1 :].max()
            for line in ax.get_lines()
        )
        pad = 0.05 * (top - bottom)
        ax.set_ylim(bottom - pad, top + pad)
    ax.grid(lw=0.4)
    ax.tick_params(length=2, pad=1.5)


def grid(
    df: pd.DataFrame, rows: list[list[str]], path: str, height: float, labels: list[str]
):
    """Panels in rows with one legend below the grid."""
    colors = dict(zip(labels, plt.cm.tab10.colors))
    fig, axes = plt.subplots(
        len(rows), len(rows[0]), figsize=(6.9, height), squeeze=False
    )
    for row_axes, targets in zip(axes, rows):
        for ax, target in zip(row_axes, targets):
            panel(ax, df[df["target_fn"] == target], target, colors)
    for ax in axes[-1]:
        ax.set_xlabel("Acquisitions", labelpad=1)

    handles = [plt.Line2D([], [], color=colors[m], lw=1.0, label=m) for m in labels]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(handles),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.16 / height, 1, 1), w_pad=0.4, h_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def kernel_profile_runs() -> pd.DataFrame:
    """Headline method at each target's selected lengthscale, one pseudo-method per GP profile."""
    best = read(SUMMARY, columns=["target_fn", "lengthscale"])
    best = best.assign(lengthscale=pd.to_numeric(best["lengthscale"]))
    best = best.groupby("target_fn")["lengthscale"].first().reset_index()
    df = read(YS_ALL, filters=[("method", "==", "ours_no_natural_grad")])
    df = df.assign(lengthscale=pd.to_numeric(df["lengthscale"]))
    df = df.merge(best, on=["target_fn", "lengthscale"])
    return df.assign(method=df["profile"].map(PROFILES))


def read(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_parquet(path, **kwargs)
    return df.astype({c: "object" for c in df.select_dtypes("category")})


if __name__ == "__main__":
    rows = [RIDGES, REALWORLD, SINC]
    height = 4.75
    ys = read(YS)
    for name, methods in ABLATIONS.items():
        df = filter_and_rename_methods(ys, methods)
        grid(df, rows, f"{FIGURES}/{name}.pdf", height, list(methods.values()))
    grid(
        kernel_profile_runs(),
        rows,
        f"{FIGURES}/kernel_profile_ablation.pdf",
        height,
        list(PROFILES.values()),
    )
