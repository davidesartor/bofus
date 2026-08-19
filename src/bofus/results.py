"""Read the converted results archive back into the package's own types."""

import pickle
from pathlib import Path

import jax.numpy as jnp

from . import rkhs

RESULTS_DIR = "results_converted"


def decode_location(entry: dict) -> rkhs.Function | rkhs.BernsteinPolynomial:
    """Rebuild one candidate from the tagged arrays written by the converter."""
    if entry["kind"] == "bernstein":
        return rkhs.BernsteinPolynomial(c=jnp.asarray(entry["c"]))

    if entry["kind"] == "rkhs_function":
        assert entry["metric"] == "Euclidean", entry["metric"]
        assert entry["profile"] == "SquaredExponential", entry["profile"]
        return rkhs.Function(
            rho=jnp.asarray(entry["rho"]),
            x=jnp.asarray(entry["x"]),
            a=jnp.asarray(entry["a"]),
        )

    raise ValueError(f"unhandled location kind {entry['kind']}")


def load_result(path: str | Path) -> dict:
    """One run, with candidates as Function objects.

    Files the sweep wrote after the conversion pass are already in the current
    format, so they come back untouched.
    """
    with open(path, "rb") as f:
        result = pickle.load(f)

    if "format_version" not in result:
        return result

    return result | dict(
        observation_locations=[
            decode_location(entry) for entry in result["observation_locations"]
        ]
    )
