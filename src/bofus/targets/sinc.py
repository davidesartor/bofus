from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class TestFunction(Protocol):
    d: int  # dimension of the input space
    m: int = 1  # number of outputs

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


class SincProjection(TestFunction):
    """L2 distance to a sinc target: int (f - t)^2.

    As for Ridge, n_points=None integrates on a tensor grid, a finite n_points is the
    Monte Carlo estimate on n uniform points and converges to the same functional.
    """

    def __init__(self, d: int = 1, n_points: int | None = None, seed: int = 0):
        self.d = d
        self.n_points = n_points
        if n_points is None:
            n = max(round(4096 ** (1 / d)), 2)
            axis = (jnp.arange(n) + 0.5) / n
            grid = jnp.meshgrid(*[axis] * d, indexing="ij")
            self.x = jnp.stack(grid, axis=-1).reshape(-1, d)
        else:
            self.x = jr.uniform(jr.key(seed), (n_points, d))

    @eqx.filter_jit
    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        target = jnp.sinc(2 * jnp.pi * self.x - jnp.pi).mean(axis=-1)
        pred = jax.vmap(f)(self.x)
        return jnp.mean(jnp.square(pred - target))
