from typing import Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from vlse.functions.base import TestFunction as Profile



class Projection:
    """L2 distance between f and the profile min-max rescaled to [-1, 1], via quadrature."""

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
    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        """Quadrature estimate of ||tanh(f) - profile||_L2 on the unit box."""
        # quadrature points live on the unit box, f on [-1, 1]
        fx = jnp.tanh(jax.vmap(lambda x: f(x, **f_kwargs))(2 * self.x - 1)[:, 0])
        return jnp.sqrt(jnp.mean(jnp.square(fx - self.y)))
