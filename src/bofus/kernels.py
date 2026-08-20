from typing import Callable, Optional
from jaxtyping import Array, Float

import jax.numpy as jnp

Profile = Callable[[Float[Array, "..."]], Float[Array, "..."]]
"""Kernel profile as a function of squared distance, patched at zero with its 1st order taylor."""


def squared_exponential(d2: Float[Array, "..."]) -> Float[Array, "..."]:
    return jnp.exp(-0.5 * d2)


def matern12(d2: Float[Array, "..."]) -> Float[Array, "..."]:
    r = jnp.sqrt(jnp.where(d2 > 0, d2, 1.0))
    k = jnp.exp(-r)
    taylor = 1.0  # k = 1 - sqrt(d2) + ..., a half power: nothing to expand to
    return jnp.where(d2 > 0, k, taylor)


def matern32(d2: Float[Array, "..."]) -> Float[Array, "..."]:
    r = jnp.sqrt(3 * jnp.where(d2 > 0, d2, 1.0))
    k = (1 + r) * jnp.exp(-r)
    taylor = 1.0 - 1.5 * d2  # k(0) + k'(0) d2, k'(0) = -3/2
    return jnp.where(d2 > 0, k, taylor)


def matern52(d2: Float[Array, "..."]) -> Float[Array, "..."]:
    r = jnp.sqrt(5 * jnp.where(d2 > 0, d2, 1.0))
    k = (1 + r + r**2 / 3) * jnp.exp(-r)
    taylor = 1.0 - 5 / 6 * d2  # k(0) + k'(0) d2, k'(0) = -5/6
    return jnp.where(d2 > 0, k, taylor)


def sq_euclidean(
    rho: Float[Array, "#d"],
    x1: Float[Array, "n d"],
    x2: Optional[Float[Array, "m d"]] = None,
) -> Float[Array, "n m"]:
    """Pairwise scaled squared euclidean distances, exactly zero on a self block's diagonal."""
    z1 = x1 / rho
    sqn1 = jnp.sum(z1**2, axis=-1)
    if x2 is not None:
        z2 = x2 / rho
        sqn2 = jnp.sum(z2**2, axis=-1)
        d2 = sqn1[:, None] + sqn2[None, :] - 2 * z1 @ z2.T
    else:
        G = z1 @ z1.T
        sqn = jnp.diag(G)
        d2 = sqn[:, None] + sqn[None, :] - 2 * G
        d2 = jnp.fill_diagonal(d2, 0.0, inplace=False)
    d2 = jnp.maximum(d2, 0.0)
    return d2
