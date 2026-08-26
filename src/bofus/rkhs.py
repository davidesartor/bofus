from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
from .kernels import rbf, euclidean_distance


class RBFMixture(NamedTuple):
    l: Float[Array, "... m d"]
    x: Float[Array, "... m d"]
    a: Float[Array, "... m"]

    @jax.jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        """Evaluate the function at points x."""
        x = x[..., *(None,) * self.a.ndim, :]
        d = euclidean_distance(self.l, x, self.x)
        y = jnp.sum(self.a * rbf(d), axis=-1)
        return y

    @staticmethod
    @jax.jit
    def from_lxy(
        l: Float[Array, "... m d"],
        x: Float[Array, "... m d"],
        y: Float[Array, "... m"],
        eps: Scalar = jnp.array(1e-2),
    ):
        """Construct a mixture of RBFs that interpolates the given points."""
        d = euclidean_distance(
            l[..., None, :, :],
            x[..., :, None, :],
            x[..., None, :, :],
        )
        # zero out the diagonal, float arithmetic does not cancel out exactly
        d = jnp.where(jnp.eye(y.shape[-1], dtype=bool), 0.0, d)
        Kxx = rbf(d) + eps * jnp.eye(y.shape[-1])
        a = jnp.linalg.solve(Kxx, y[..., None]).squeeze(-1)
        return RBFMixture(l=l, x=x, a=a)

    def pad_to(self, m: int) -> Self:
        """Pad the mixture to a larger number of basis points."""
        *b, m0, d = self.l.shape
        l = jnp.ones((*b, m, d)).at[..., :m0, :].set(self.l)
        x = jnp.zeros((*b, m, d)).at[..., :m0, :].set(self.x)
        a = jnp.zeros((*b, m)).at[..., :m0].set(self.a)
        return self._replace(l=l, x=x, a=a)


@jax.jit
def rbf_inner(
    l0: Float[Array, "... d"],
    f1: RBFMixture,
    f2: RBFMixture,
) -> Float[Array, "..."]:
    """Inner product of RBF mixtures, seen in a wider RBF rkhs."""
    # broadcast to common shape (..., m1, m2, d)
    l1, x1 = f1.l[..., :, None, :], f1.x[..., :, None, :]
    l2, x2 = f2.l[..., None, :, :], f2.x[..., None, :, :]
    l0 = l0[..., None, None, :]

    # lengthscale factors, patched where the pair does not fit the ambient rkhs
    ls = l1 + l2 - l0
    valid = jnp.all(ls > 0.0, axis=(-3, -2, -1))
    ls = jnp.where(ls > 0.0, ls, 1.0)
    lp = l1 * l2 / (l0 * ls)
    scale = jnp.sqrt(jnp.prod(lp, axis=-1))

    # inner kernel matrix, infinite if any pair falls outside the ambient rkhs
    K = scale * rbf(euclidean_distance(ls, x1, x2))
    k = jnp.einsum("...ij,...i,...j->...", K, f1.a, f2.a)
    return jnp.where(valid, k, jnp.inf)


@jax.jit
def rbf_distance(
    l0: Float[Array, "... d"],
    f1: RBFMixture,
    f2: RBFMixture,
) -> Float[Array, "..."]:
    """Distance between RBF mixtures, seen in a wider RBF rkhs."""
    sqn1 = rbf_inner(l0, f1, f1)
    sqn2 = rbf_inner(l0, f2, f2)
    cross = rbf_inner(l0, f1, f2)
    d2 = sqn1 + sqn2 - 2.0 * cross
    # fix nan and inf gradients returning the subgradient instead
    d2 = jnp.nan_to_num(d2, nan=jnp.inf, posinf=jnp.inf)
    d = jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0))
    return jnp.where(d2 > 0.0, d, 0.0)
