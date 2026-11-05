from typing import Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from vlse.functions.base import TestFunction as Profile

from targets import evaluate_on


class Projection:
    """L2 distance between f and a rescaled profile on the unit box, via quadrature.

    The profile is min-max rescaled to [-1, 1], the closure of the range of f.
    With n_points=None uses a midpoint tensor grid, otherwise a Monte Carlo sample.
    RBF mixtures are dense in L2, so the infimum of the objective is 0.
    """

    def __init__(self, profile: Profile, n_points: int | None = None, seed: int = 0):
        self.d = d = profile.d
        self.k = 1
        self.profile = profile
        self.optimum = 0.0

        if n_points is None:
            # tensor grid with total kept around 4096
            n = max(round(4096 ** (1 / d)), 2)
            axis = (jnp.arange(n) + 0.5) / n
            grid = jnp.meshgrid(*[axis] * d, indexing="ij")
            self.x = jnp.stack(grid, axis=-1).reshape(-1, d)
        else:
            self.x = jr.uniform(jr.key(seed), (n_points, d), minval=0.0, maxval=1.0)

        # min-max rescale the profile to [-1, 1] over the quadrature points
        y = jax.vmap(profile)(self.x)
        self.shift, self.scale = (y.max() + y.min()) / 2, (y.max() - y.min()) / 2
        self.y = (y - self.shift) / self.scale

    @eqx.filter_jit
    def __call__(self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]]) -> Scalar:
        """Quadrature estimate of ||f - profile||_L2 on the unit box."""
        fx = evaluate_on(f, self.x, self.k)[:, 0]
        return jnp.sqrt(jnp.mean(jnp.square(fx - self.y)))
