"""LaTeX tables of the paper, written from the summary tables of the sweep."""

import os

import numpy as np
import pandas as pd

from bofus.plots import (
    RESULTS_DIR,
    filter_and_rename_methods,
    filter_best_profile_per_lengthscale,
    median_and_ci,
)

TABLES_DIR = os.environ.get("TABLES_DIR", "latex/neurips/tables")

NAMES = {
    "gramacylee": "GramacyLee",
    "ackley": "Ackley",
    "hartmann": "Hartmann",
    "rosenbrock": "Rosenbrock",
    "michalewicz": "Michalewicz",
    "brachistochrone": "Brachistochrone",
    "mnist": "MNIST",
    "pendulum": "Pendulum",
    "pinwheel": "Pinwheel",
    "hopper": "Hopper",
    "sinc1d": "Sinc-1D",
    "sinc2d": "Sinc-2D",
    "sinc3d": "Sinc-3D",
    "sinc4d": "Sinc-4D",
    "sinc5d": "Sinc-5D",
}
RIDGES = ["gramacylee", "ackley", "hartmann", "rosenbrock", "michalewicz"]
REALWORLD = ["brachistochrone", "mnist", "pendulum", "pinwheel", "hopper"]
SINC = ["sinc1d", "sinc2d", "sinc3d", "sinc4d", "sinc5d"]
MAIN_TARGETS = RIDGES + REALWORLD
METRICS = {"avg_regret": "Avg", "best_y": "Final"}
METHODS = {
    "random": "Random",
    "kundu": r"\kundu",
    "vien": r"\vien",
    "shilton": r"\shilton",
    "vellanky": r"\vellanky",
    "ours_no_natural_grad": r"\wycoff",
}
# power of ten each regret table row is displayed in
EXPONENTS = {
    "gramacylee": -1,
    "ackley": 1,
    "hartmann": 0,
    "rosenbrock": 4,
    "michalewicz": 0,
    "brachistochrone": -2,
    "mnist": -2,
    "pendulum": 2,
    "pinwheel": 0,
    "hopper": 0,
    "sinc1d": -2,
    "sinc2d": -2,
    "sinc3d": -2,
    "sinc4d": -2,
    "sinc5d": -2,
}
# targets whose regrets are all negative, shown with the sign folded into the unit
NEGATED = {"hopper"}
SMOOTHNESS = {
    "rbf": r"$\mathcal{C}(\infty)$",
    "matern52": r"$\mathcal{C}(2)$",
    "matern32": r"$\mathcal{C}(1)$",
    "matern12": r"$\mathcal{C}(0)$",
}
NOT_SUPPORTED = r"\text{\ding{55}}"


def load(name: str) -> pd.DataFrame:
    """Summary table with plain object labels and a numeric lengthscale."""
    df = pd.read_parquet(f"{RESULTS_DIR}/{name}.parquet")
    df = df.astype({c: "object" for c in df.select_dtypes("category")})
    return df.assign(lengthscale=pd.to_numeric(df["lengthscale"]))


def fmt(v: float) -> str:
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def scale(target: str) -> float:
    return (-1.0 if target in NEGATED else 1.0) * 10.0 ** EXPONENTS[target]


def math_cell(text: str, bold: bool = False) -> str:
    return rf"$\mathbf{{{text}}}$" if bold else f"${text}$"


def float_env(caption: str, label: str, body: list[str]) -> str:
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            *body,
            r"\end{table}",
            "",
        ]
    )


def tabular(columns: str, lines: list[str], fit_width: bool = False) -> list[str]:
    """Booktabs tabular in the tiny, tight style shared by every table of the paper."""
    # fit_width spreads the columns over the text width without scaling the font
    begin, end = rf"\begin{{tabular}}{{{columns}}}", r"\end{tabular}"
    if fit_width:
        begin = rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} {columns}}}"
        end = r"\end{tabular*}"
    return [
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\tiny",
        begin,
        r"\toprule",
        *lines,
        r"\bottomrule",
        end,
    ]


def method_header(first: str, methods: dict[str, str], metrics: list[str]) -> list[str]:
    """Method names spanning one column per metric, with the metric names below."""
    span = len(metrics)
    names = " & ".join(rf"\multicolumn{{{span}}}{{c}}{{{m}}}" for m in methods.values())
    rules = " ".join(
        rf"\cmidrule(lr){{{2 + span * i}-{1 + span * (i + 1)}}}"
        for i in range(len(methods))
    )
    subheader = " & ".join(rf"\textbf{{{m}}}" for m in metrics * len(methods))
    return [
        rf"{first} & {names} \\",
        rules,
        rf" & {subheader} \\",
        r"\midrule",
    ]


def cell_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Median and 95% CI of metric per (target, method)."""
    rows = [
        (target, method, *median_and_ci(g[metric].to_numpy()))
        for (target, method), g in df.groupby(["target_fn", "method"])
    ]
    return pd.DataFrame(
        rows, columns=["target_fn", "method", "median", "lower", "upper"]
    )


def regret_block(stats: pd.DataFrame, targets: list[str]) -> list[str]:
    """One regret metric of every method on a row of targets."""
    header = " & ".join(
        rf"\textbf{{{NAMES[t]}}} [${'-' * (t in NEGATED)}10^{{{EXPONENTS[t]}}}$]"
        for t in targets
    )
    lines = [rf"\textbf{{Method}} & {header} \\", r"\midrule"]
    for method, label in METHODS.items():
        cells = []
        for target in targets:
            if (target, method) not in stats.index:
                cells.append(NOT_SUPPORTED)
                continue
            cell = stats.loc[target, method]
            median, lower, upper = cell / scale(target)
            lower, upper = sorted([lower, upper])
            is_best = cell["median"] == stats.loc[target]["median"].min()
            cells.append(
                math_cell(rf"{fmt(median)}\ [{fmt(lower)}, {fmt(upper)}]", is_best)
            )
        lines.append(" & ".join([f"{label:<10}", *cells]) + r" \\")
    return lines


def regret_table(df: pd.DataFrame, rows: list[list[str]], metric: str) -> str:
    """One regret metric for every method, one block per row of targets."""
    df = df[df["method"].isin(METHODS)]
    stats = cell_stats(df, metric).set_index(["target_fn", "method"])
    lines = []
    for i, targets in enumerate(rows):
        if i:
            lines.append(r"\midrule\midrule")
        lines += regret_block(stats, targets)
    name = {"avg_regret": "Average", "best_y": "Final"}[metric]
    return float_env(
        rf"{name} regret (median and 95\% CI) across methods and benchmark functions, "
        r"one row per function class. $\text{\ding{55}}$ indicates the method does not support the corresponding problem class.",
        f"tab:regret_{name.lower()}",
        tabular("l " + "c " * len(rows[0]), lines),
    )


def tolerant_ranks(stats: pd.DataFrame) -> pd.Series:
    """Rank per (target, method); a method ties with any better one whose median its own CI covers."""
    ranks = {}
    for target, g in stats.groupby("target_fn"):
        for row in g.itertuples():
            better = g[(g["median"] < row.median)]
            beaten = better[~better["median"].between(row.lower, row.upper)]
            ranks[target, row.method] = 1 + len(beaten)
    return pd.Series(ranks)


def rank_cells(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean tolerant rank of every method, per metric, over the targets it ran on."""
    return {
        metric: tolerant_ranks(cell_stats(df, metric)).groupby(level=1).mean().to_dict()
        for metric in METRICS
    }


def rank_row(label: str, ranks: dict[str, dict[str, float]]) -> str:
    best = {m: min(ranks[m].values()) for m in METRICS}
    cells = []
    for method in METHODS:
        for metric in METRICS:
            r = ranks[metric].get(method)
            cells.append(
                NOT_SUPPORTED if r is None else math_cell(f"{r:.2f}", r == best[metric])
            )
    return f"{label} & " + " & ".join(cells) + r" \\"


def median_rows(df: pd.DataFrame, targets: list[str]) -> list[str]:
    """Per-target median rows with a shared power of ten, rank-1 (tied) cells in bold."""
    stats = {m: cell_stats(df, m).set_index(["target_fn", "method"]) for m in METRICS}
    ranks = {m: tolerant_ranks(stats[m].reset_index()) for m in METRICS}
    lines = []
    for target in targets:
        largest = max(abs(stats[m].loc[target]["median"]).max() for m in METRICS)
        exponent = int(np.floor(np.log10(largest)))
        sign = "-" if target in NEGATED else ""
        unit = (-1.0 if target in NEGATED else 1.0) * 10.0**exponent
        cells = []
        for method in METHODS:
            for metric in METRICS:
                if (target, method) not in stats[metric].index:
                    cells.append(NOT_SUPPORTED)
                    continue
                median = stats[metric].loc[(target, method), "median"]
                cells.append(
                    math_cell(
                        fmt(median / unit),
                        ranks[metric][target, method] == 1,
                    )
                )
        lines.append(
            rf"{NAMES[target]} [${sign}10^{{{exponent}}}$] & "
            + " & ".join(cells)
            + r" \\"
        )
    return lines


def regret_medians_tabular(lines: list[str]) -> list[str]:
    return tabular("l " + "cc " * len(METHODS), lines, fit_width=True)


def best_hyperparameters(summary_filtered: pd.DataFrame) -> str:
    methods = {k: v for k, v in METHODS.items() if k not in ("random", "vellanky")}
    lines = [
        r"\textbf{Benchmark} & \textbf{\gls{rkhs} scale} & "
        + rf"\multicolumn{{{len(methods)}}}{{c}}{{\textbf{{Surrogate kernel}}}} \\",
        rf"\cmidrule(lr){{3-{2 + len(methods)}}}",
        " & & " + " & ".join(methods.values()) + r" \\",
    ]
    combos = summary_filtered.groupby(["target_fn", "method"])[
        ["lengthscale", "profile"]
    ].first()
    for group in (RIDGES, REALWORLD, SINC):
        lines.append(r"\midrule")
        for target in group:
            scale = combos.loc[(target, "ours_no_natural_grad"), "lengthscale"]
            profiles = [SMOOTHNESS[combos.loc[(target, m), "profile"]] for m in methods]
            lines.append(
                rf"{NAMES[target]} & ${scale:g}$ & " + " & ".join(profiles) + r" \\"
            )
    return float_env(
        r"Selected hyperparameters per method and benchmark. "
        r"The \gls{gp} kernel determines the order of differentiability $\mathcal{C}(\nu)$ of the surrogate model ($\mathcal{C}(\infty)$ = RBF). "
        r"The \gls{rkhs} lengthscale $\ell$ is shared by all methods on a benchmark.",
        "tab:hyperparams",
        tabular("l c " + "c" * len(methods), lines, fit_width=True),
    )


def timings(summary_filtered: pd.DataFrame) -> str:
    methods = {k: v for k, v in METHODS.items() if k != "random"}
    df = filter_and_rename_methods(summary_filtered, methods)
    lines = method_header(r"\textbf{Benchmark}", methods, ["Fit", "Acq."])
    for target in NAMES:
        cells = []
        for method in methods.values():
            g = df[(df["target_fn"] == target) & (df["method"] == method)]
            cells += (
                [NOT_SUPPORTED] * 2
                if g.empty
                else [
                    math_cell(f"{np.median(g['t_fit']):.0f}"),
                    math_cell(f"{np.median(g['t_acq']):.0f}"),
                ]
            )
        lines.append(f"{NAMES[target]} & " + " & ".join(cells) + r" \\")
    return float_env(
        r"Median total wall-clock time per run (seconds) spent fitting the surrogate and optimizing the acquisition, "
        r"over the $110$-evaluation budget, as recorded in the main sweep on a heterogeneous cluster. "
        r"Times include \textsc{jax} compilation and are only indicative across rows.",
        "tab:timings_cluster",
        tabular("l " + "rr " * len(methods), lines, fit_width=True),
    )


def lengthscale_tables(summary_all: pd.DataFrame) -> tuple[str, str]:
    """Mean ranks and full medians at every lengthscale, each method at its own best profile there."""
    per_scale = filter_best_profile_per_lengthscale(
        filter_and_rename_methods(
            summary_all[summary_all["target_fn"].isin(MAIN_TARGETS)],
            dict(zip(METHODS, METHODS)),
        )
    )
    scales = sorted(per_scale["lengthscale"].unique())
    metrics = list(METRICS.values())
    rank_lines = method_header(r"$\ell$", METHODS, metrics) + [
        rank_row(f"${s:g}$", rank_cells(per_scale[per_scale["lengthscale"] == s]))
        for s in scales
    ]
    full_lines = method_header(r"\textbf{Benchmark}", METHODS, metrics)
    for i, s in enumerate(scales):
        at_scale = per_scale[per_scale["lengthscale"] == s]
        if i:
            full_lines.append(r"\midrule\midrule")
        full_lines.append(
            rf"\multicolumn{{{1 + 2 * len(METHODS)}}}{{l}}{{\textit{{\gls{{rkhs}} lengthscale $\ell = {s:g}$}}}} \\"
        )
        full_lines += median_rows(at_scale, MAIN_TARGETS)
        full_lines.append(rank_row(r"\textit{mean rank}", rank_cells(at_scale)))

    ranks = float_env(
        r"Mean rank over the ten main benchmarks at each \gls{rkhs} lengthscale $\ell$, each method using its own best \gls{gp} profile at that $\ell$. "
        r"A method ties with any better one whose median lies within its own 95\% CI. Lower is better; best per row in bold.",
        "tab:lengthscale_ranks",
        regret_medians_tabular(rank_lines),
    )
    full = float_env(
        r"Median final and average regret at every \gls{rkhs} lengthscale, each method at its own best \gls{gp} profile. "
        r"Each benchmark carries a common power of ten. Methods tied for first (see \cref{tab:lengthscale_ranks}) in bold.",
        "tab:lengthscale_full",
        regret_medians_tabular(full_lines),
    )
    return ranks, full


def matern52(summary_all: pd.DataFrame, summary_filtered: pd.DataFrame) -> str:
    """The selected lengthscale of each target, with one Matern-5/2 profile for every method."""
    selected = (
        summary_filtered.groupby("target_fn")["lengthscale"].first().rename("selected")
    )
    fixed = summary_all.merge(selected, left_on="target_fn", right_index=True)
    fixed = fixed[
        (fixed["lengthscale"] == fixed["selected"]) & (fixed["profile"] == "matern52")
    ]
    fixed = fixed[fixed["target_fn"].isin(MAIN_TARGETS) & fixed["method"].isin(METHODS)]
    lines = (
        method_header(r"\textbf{Benchmark}", METHODS, list(METRICS.values()))
        + median_rows(fixed, MAIN_TARGETS)
        + [r"\midrule", rank_row(r"\textit{mean rank}", rank_cells(fixed))]
    )
    return float_env(
        r"Median final and average regret with the Mat\'ern-5/2 \gls{gp} profile for every method, at the selected \gls{rkhs} lengthscale. "
        r"Methods tied for first in bold.",
        "tab:matern52",
        regret_medians_tabular(lines),
    )


if __name__ == "__main__":
    summary_all = load("summary_all")
    summary_filtered = load("summary_filtered")
    lengthscale_ranks, lengthscale_full = lengthscale_tables(summary_all)
    tables = {
        "results_avg_ci": regret_table(
            summary_filtered, [SINC, RIDGES, REALWORLD], "avg_regret"
        ),
        "results_final_ci": regret_table(
            summary_filtered, [SINC, RIDGES, REALWORLD], "best_y"
        ),
        "best_lengthscales": best_hyperparameters(summary_filtered),
        "timings_cluster": timings(summary_filtered),
        "lengthscale_ranks": lengthscale_ranks,
        "lengthscale_full": lengthscale_full,
        "matern52": matern52(summary_all, summary_filtered),
    }
    os.makedirs(TABLES_DIR, exist_ok=True)
    for name, tex in tables.items():
        with open(f"{TABLES_DIR}/{name}.tex", "w") as f:
            f.write(tex)
