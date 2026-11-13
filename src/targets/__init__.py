from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar


class TestFunction(Protocol):
    """Finite functional objective to minimize over f: [-1, 1]^d -> R^k."""

    d: int  # number of inputs
    k: int  # number of outputs
    optimum: float | None  # infimum when known, for regret

    def __call__(self, f: Callable[..., Float[Array, "o"]], **f_kwargs) -> Scalar:
        """Objective value of f, evaluated as f(x, **f_kwargs)."""
        ...


from .ridge import Ridge
from .projection import Projection
