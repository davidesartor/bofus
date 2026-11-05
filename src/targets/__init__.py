from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import jax


class TestFunction(Protocol):
    """Functional objective to minimize over f: [0, 1]^d -> R^k.

    Conventions shared by the targets:
    - f is called pointwise on the unit cube and returns k outputs in (-1, 1),
      squashed by tanh; each target maps that range to its own units.
    - the returned value is finite, so the surrogate never sees nan or inf.
    - optimum is the infimum of the objective when known (None otherwise), so
      results can be reported as regret y - optimum.
    """

    d: int  # number of inputs
    k: int  # number of outputs
    optimum: float | None

    def __call__(
        self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]]
    ) -> Scalar: ...


def evaluate_on(
    f: Callable[[Float[Array, "d"]], Float[Array, "k"]],
    x: Float[Array, "n d"],
    k: int,
) -> Float[Array, "n k"]:
    """Evaluate f at the points x, checking it returns the k outputs the target expects."""
    y = jax.vmap(f)(x)
    if y.shape != (x.shape[0], k):
        raise ValueError(f"f should return {k} outputs per point, got shape {y.shape[1:]}")
    return y


from .ridge import Ridge
from .projection import Projection
from .classic_control import ClassicControl
from .mujoco import Mujoco
