from typing import Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from vlse.functions.base import TestFunction as Profile


class Projection:
    """L2 distance between f and a profile on the unit box, via quadrature.

    With n_points=None uses a midpoint tensor grid, otherwise a Monte Carlo sample.
    """

    def __init__(self, profile: Profile, n_points: int | None = None, seed: int = 0):
        self.d = d = profile.d
        self.k = 1
        self.profile = profile
        if n_points is None:
            # tensor grid with total kept around 4096
            n = max(round(4096 ** (1 / d)), 2)
            axis = (jnp.arange(n) + 0.5) / n
            grid = jnp.meshgrid(*[axis] * d, indexing="ij")
            self.x = jnp.stack(grid, axis=-1).reshape(-1, d)
        else:
            self.x = jr.uniform(jr.key(seed), (n_points, d), minval=0.0, maxval=1.0)
        self.y = jax.vmap(profile)(self.x)

    @eqx.filter_jit
    def __call__(self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]]) -> Scalar:
        """Quadrature estimate of ||f - profile||_L2 on the unit box."""
        return jnp.sqrt(jnp.mean(jnp.square(jax.vmap(f)(self.x).squeeze(-1) - self.y)))
