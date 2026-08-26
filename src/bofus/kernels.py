from typing import Callable
from jaxtyping import Array, Float

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

Profile = Callable[[Float[Array, "..."]], Float[Array, "..."]]


@jax.jit
def euclidean_distance(
    l: Float[Array, "... d"],
    x1: Float[Array, "... d"],
    x2: Float[Array, "... d"],
) -> Float[Array, "..."]:
    """Euclidean distance scaled by squared lengthscales l."""
    d2 = jnp.sum(jnp.square(x1 - x2) / l, axis=-1)
    d = jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0))
    d = jnp.where(d2 > 0.0, d, 0.0)
    return d


@jax.jit
def rbf(d: Float[Array, "..."]) -> Float[Array, "..."]:
    """Radial basis function kernel."""
    inf = jnp.isinf(d)
    d = jnp.where(inf, 1.0, d)
    k = jnp.exp(-0.5 * d**2)
    return jnp.where(inf, 0.0, k)


@jax.jit
def matern12(d: Float[Array, "..."]) -> Float[Array, "..."]:
    """Matern 1/2 kernel."""
    inf = jnp.isinf(d)
    d = jnp.where(inf, 1.0, d)
    k = jnp.exp(-d)
    return jnp.where(inf, 0.0, k)


@jax.jit
def matern32(d: Float[Array, "..."]) -> Float[Array, "..."]:
    """Matern 3/2 kernel."""
    inf = jnp.isinf(d)
    d = jnp.where(inf, 1.0, d)
    k = (1 + jnp.sqrt(3.0) * d) * jnp.exp(-jnp.sqrt(3.0) * d)
    return jnp.where(inf, 0.0, k)


@jax.jit
def matern52(d: Float[Array, "..."]) -> Float[Array, "..."]:
    """Matern 5/2 kernel."""
    inf = jnp.isinf(d)
    d = jnp.where(inf, 1.0, d)
    k = (1 + jnp.sqrt(5.0) * d + 5.0 / 3.0 * d**2) * jnp.exp(-jnp.sqrt(5.0) * d)
    return jnp.where(inf, 0.0, k)


def matern(p: int):
    @jax.jit
    def profile(d: Float[Array, "..."]) -> Float[Array, "..."]:
        """Matern kernel with half-integer smoothness p + 1/2."""
        inf = jnp.isinf(d)
        d = jnp.where(inf, 1.0, d)
        r = jnp.sqrt(2.0 * p + 1.0) * d

        # sum terms fully in log space, coefficients and powers overflow separately for large p
        log_coef = lambda i: (
            gammaln(p + 1)
            - gammaln(2 * p + 1)
            + gammaln(p + i + 1)
            - gammaln(i + 1)
            - gammaln(p - i + 1)
        )
        zero = r == 0.0
        log_2r = jnp.log(jnp.where(zero, 1.0, 2.0 * r))
        k = sum(jnp.exp(log_coef(i) + (p - i) * log_2r - r) for i in range(p + 1))
        k = jnp.where(zero, 1.0, k)
        return jnp.where(inf, 0.0, k)

    return profile
