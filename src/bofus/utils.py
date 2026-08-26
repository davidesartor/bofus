import math
from jaxtyping import Array, Bool, Float, Key

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


@jax.jit
def mask_covariance(
    cov: Float[Array, "... n n"],
    mask: Bool[Array, "... n"],
) -> Float[Array, "... n n"]:
    """Zero the masked-out rows/columns, an identity block keeps the factorization valid."""
    # zero out
    I = jnp.eye(cov.shape[-1], dtype=cov.dtype)
    cov = cov * mask[..., :, None] * mask[..., None, :]
    cov = cov + I * (1.0 - mask[..., None, :])
    return cov


@eqx.filter_jit
def latin_hypercube_sample(
    key: Key,
    shape: tuple[int, ...],
) -> Float[Array, "..."]:
    """Sample shape[0] unit-cube points, stratified over the flattened trailing dims."""
    n, *rest = shape
    dim = math.prod(rest)
    perm_key, u_key = jr.split(key)
    perm = jax.vmap(lambda k: jr.permutation(k, n))(jr.split(perm_key, dim))
    u = jr.uniform(u_key, (n, dim))
    x = (perm.T + u) / n
    return x.reshape(shape)


@eqx.filter_jit
def rescale(
    x: Float[Array, "..."],
    low: Float[Array, "..."],
    high: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Map unit-cube points to the box [low, high]."""
    return low + x * (high - low)
