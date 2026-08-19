import pickle
from pathlib import Path

import numpy as np
import pytest

REPRO_OUT = Path("scripts/repro_out")
ARCHIVE_ROOT = Path("results/neurips")
METHODS = ["random", "ours", "vien", "kundu", "shilton", "vellanky"]
PROFILES = ["rbf", "matern52", "matern32", "matern12"]


def observation_values(path: Path):
    with path.open("rb") as fh:
        return np.asarray(pickle.load(fh)["observation_values"])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("profile", PROFILES)
def test_reproduction_matches_head_and_archive(method, profile):
    """A head-vs-new repro run (see scripts/repro_test.sbatch) must match itself and the archive.

    Skipped unless scripts/repro_out/ has been populated by that sbatch job.
    """
    head = REPRO_OUT / f"head_{method}_{profile}.pkl"
    new = REPRO_OUT / f"new_{method}_{profile}.pkl"
    archive = (
        ARCHIVE_ROOT / "sinc1d" / method / f"{profile}_lengthscale_0.2" / "seed_0.pkl"
    )
    if not (head.exists() and new.exists() and archive.exists()):
        pytest.skip("repro_out/ or archived results not present")

    np.testing.assert_array_equal(observation_values(new), observation_values(head))
    np.testing.assert_array_equal(observation_values(new), observation_values(archive))
