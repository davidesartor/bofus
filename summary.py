from itertools import product
import argparse
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/neurips")
LABELS = ["target_fn", "method", "profile", "lengthscale"]

# the lengthscale is picked on the methods the paper compares, so an ablation variant
# cannot drag the whole sweep onto a lengthscale none of the main methods run at
MAIN_METHODS = ["vien", "shilton", "vellanky", "kundu", "ours_no_natural_grad"]


def filter_best_lengthscale(df: pd.DataFrame) -> pd.DataFrame:
    # best lengthscale per target_fn across the (method, profile) of the main variants
    best_lengthscale = (
        df[df["method"].isin(MAIN_METHODS)]
        .groupby(["target_fn", "method", "profile", "lengthscale"], observed=True)[
            "best_y"
        ]
        .median()
        .reset_index()
        .loc[lambda d: d.groupby("target_fn", observed=True)["best_y"].idxmin()][
            ["target_fn", "lengthscale"]
        ]
    )
    return df.merge(best_lengthscale, on=["target_fn", "lengthscale"])


def filter_best_profile(df: pd.DataFrame) -> pd.DataFrame:
    # best profile per (target_fn, method)
    best_profile = (
        df.groupby(["target_fn", "method", "profile"])["best_y"]
        .median()
        .reset_index()
        .loc[lambda d: d.groupby(["target_fn", "method"])["best_y"].idxmin()][
            ["target_fn", "method", "profile"]
        ]
    )
    return df.merge(best_profile, on=["target_fn", "method", "profile"])


def config_path(target_fn, method, profile, lengthscale):
    return f"{RESULTS_DIR}/{target_fn}/{method}/{profile}_lengthscale_{lengthscale}/"


def read_dir(target_fn, method, profile, lengthscale):
    def read_file(f):
        seed = int(f.split("_")[1].split(".")[0])
        # incomplete/corrupted files (job killed mid-write) are skipped
        try:
            r = np.load(os.path.join(path, f), allow_pickle=True)
            y = np.minimum.accumulate(r["observation_values"])
        except (EOFError, OSError, ValueError, KeyError) as e:
            print(f"Warning: skipping {os.path.join(path, f)} ({e})")
            return None

        # dict with summary statistics
        summary = {
            "seed": seed,
            "best_y": np.min(y),
            "avg_regret": np.mean(y),
            # parquet rejects an object column, and older runs stored these as 0-d arrays
            "t_fit": float(r["surrogate_fit_time"]),
            "t_acq": float(r["acquisition_time"]),
            "t_eval": float(r["target_evaluation_time"]),
        }
        # the running best stays an array; a row per acquisition step as dicts costs
        # two orders of magnitude more memory than the floats it carries
        return summary, np.asarray(y, dtype=np.float64)

    # read all files with this combination of params
    path = config_path(target_fn, method, profile, lengthscale)
    # the sweep keeps writing while this runs, so a config dir can vanish after the scan listed it
    try:
        files = [f for f in os.listdir(path) if f.endswith(".pkl")]
    except FileNotFoundError:
        print(f"Warning: {path} disappeared during the scan")
        return None
    # files = [f for f in files if int(f.split("_")[1].split(".")[0]) < 16]
    if len(files) < 1:
        print(f"Warning: only found {len(files)} files for {path}")
        return None
    results = [r for r in map(read_file, files) if r is not None]
    if len(results) < 1:
        print(f"Warning: no readable files for {path}")
        return None
    summaries, ys = zip(*results)

    # a worker hands back arrays, not frames; the labels are the same four strings for every
    # row here, so the parent attaches them once as categoricals over the whole sweep
    step = np.concatenate([np.arange(len(y), dtype=np.int32) for y in ys])
    return list(summaries), step, np.concatenate(ys)


def labels_per_row(combos: list[tuple], counts: list[int]) -> dict[str, pd.Categorical]:
    """The four constant labels of each config, expanded to one row per result."""
    row_combo = np.repeat(np.arange(len(combos)), counts)

    columns = {}
    for name, position in [
        ("method", 1),
        ("profile", 2),
        ("target_fn", 0),
        ("lengthscale", 3),
    ]:
        categories, codes = np.unique(
            [c[position] for c in combos], return_inverse=True
        )
        columns[name] = pd.Categorical.from_codes(
            codes[row_combo], categories=categories
        )
    return columns


def parse_interval(spec: str) -> float:
    """Seconds from a "90s"/"15m"/"2h"/"1d" interval; a bare number is minutes."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    value, unit = (spec[:-1], spec[-1]) if spec[-1:] in units else (spec, "m")
    try:
        return float(value) * units[unit]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid interval {spec!r}, expected e.g. 15m"
        )


def touched_since(path: str, cutoff: float) -> bool:
    """A config counts as touched when its dir or any file in it changed after cutoff."""
    try:
        if os.stat(path).st_mtime >= cutoff:
            return True
        return any(entry.stat().st_mtime >= cutoff for entry in os.scandir(path))
    except (FileNotFoundError, NotADirectoryError):
        return False


def merge_previous(
    fresh: pd.DataFrame, path: str, keep: set[tuple[str, str, str, str]]
) -> pd.DataFrame:
    """Append the rows of a previous run for the configs that were not rescanned."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found, writing only the rescanned configs")
        return fresh
    previous = pd.read_parquet(path)
    reused = pd.MultiIndex.from_frame(previous[LABELS]).isin(keep)
    previous = previous.loc[reused, fresh.columns].copy()

    # concat falls back to strings unless both sides carry the very same categories
    for column in LABELS:
        categories = fresh[column].cat.categories.union(previous[column].cat.categories)
        fresh[column] = fresh[column].cat.set_categories(categories)
        previous[column] = previous[column].cat.set_categories(categories)
    merged = pd.concat([fresh, previous], ignore_index=True)
    return merged.assign(
        **{c: merged[c].cat.remove_unused_categories() for c in LABELS}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--update",
        type=parse_interval,
        metavar="INTERVAL",
        help="only rescan configs written in the last interval (e.g. 15m, 2h), "
        "other rows are kept from the existing table",
    )
    args = parser.parse_args()

    def listdir(path: str) -> list[str]:
        """A concurrent sweep can remove a directory between one level of the scan and the next."""
        try:
            return os.listdir(path)
        except (FileNotFoundError, NotADirectoryError):
            return []

    combos = []
    for target_fn in listdir(f"{RESULTS_DIR}/"):
        # the written tables sit next to the target dirs
        if not os.path.isdir(f"{RESULTS_DIR}/{target_fn}"):
            continue
        for method in listdir(f"{RESULTS_DIR}/{target_fn}/"):
            for profile_and_scale in listdir(f"{RESULTS_DIR}/{target_fn}/{method}/"):
                profile, _, lengthscale = profile_and_scale.split("_")
                combos.append((target_fn, method, profile, lengthscale))

    partial_update = args.update is not None
    if partial_update:
        cutoff = time.time() - args.update
        on_disk = set(combos)
        # a cold stat of every config dir is minutes of network filesystem latency, so the
        # scan waits on threads rather than one round trip at a time
        touched = joblib.Parallel(32, prefer="threads")(
            joblib.delayed(touched_since)(config_path(*c), cutoff)
            for c in tqdm(combos, desc="scanning for touched configs")
        )
        combos = [c for c, is_touched in zip(combos, touched) if is_touched]
        if not combos:
            print(f"nothing written in the last {args.update:g}s, tables left alone")
            raise SystemExit(0)
        # rows of a config that is still on disk and was not rescanned are reused as they are,
        # so a config deleted since the last run drops out instead of surviving in the old table
        reusable = on_disk - set(combos)

    # each worker returns arrays rather than frames, so the pipe carries buffers instead of
    # pickled dataframes and the parent never holds one frame per config at all
    results = joblib.Parallel(-1)(joblib.delayed(read_dir)(*c) for c in tqdm(combos))
    kept = [(c, r) for c, r in zip(combos, results) if r is not None]
    if partial_update:
        # a config caught mid-write reads as empty, so it keeps the rows of the last run
        # rather than losing them until the next scan finds it readable again
        reusable |= {c for c, r in zip(combos, results) if r is None}
    combos = [c for c, _ in kept]

    summary_df = pd.DataFrame(
        [row for _, (summaries, _, _) in kept for row in summaries]
    ).assign(**labels_per_row(combos, [len(s) for _, (s, _, _) in kept]))
    if partial_update:
        summary_df = merge_previous(
            summary_df, f"{RESULTS_DIR}/summary_all.parquet", reusable
        )
    summary_df.to_parquet(f"{RESULTS_DIR}/summary_all.parquet")

    steps = np.concatenate([step for _, (_, step, _) in kept])
    ys = np.concatenate([y for _, (_, _, y) in kept])
    ys_df = pd.DataFrame({"i": steps, "y": ys}).assign(
        **labels_per_row(combos, [len(step) for _, (_, step, _) in kept])
    )
    del steps, ys
    if partial_update:
        ys_df = merge_previous(ys_df, f"{RESULTS_DIR}/ys_all.parquet", reusable)
    ys_df.to_parquet(f"{RESULTS_DIR}/ys_all.parquet")

    # filter to only include best profile and lengthscale for each target_fn and method
    summary_df = filter_best_lengthscale(summary_df)
    summary_df = filter_best_profile(summary_df)
    summary_df.to_parquet(f"{RESULTS_DIR}/summary_filtered.parquet")

    ys_df = ys_df.merge(
        summary_df[["target_fn", "method", "profile", "lengthscale"]].drop_duplicates(),
        on=["target_fn", "method", "profile", "lengthscale"],
    )
    ys_df.to_parquet(f"{RESULTS_DIR}/ys_filtered.parquet")
