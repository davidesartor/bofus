from types import GenericAlias
from typing import NamedTuple, Optional, Self
from jaxtyping import Array, Bool, Float

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from . import utils

X_MAX = 1.0  # domain is [-X_MAX, X_MAX]^d, periodic dimensions wrap


class RBFMixture(NamedTuple):
    """RBF mixture f = sum_i a_i prod_j phi(x_j - x_ij; l_ij)."""

    l: Float[Array, "*b m d"]
    x: Float[Array, "*b m d"]
    a: Float[Array, "*b m"]

    @classmethod
    def __class_getitem__(cls, shape: str) -> GenericAlias:
        """Shape annotation RBFMixture["*b m d"], documentation only."""
        return GenericAlias(cls, shape)

    @jax.jit
    def __call__(
        self,
        x: Float[Array, "d"],
        periodic: Optional[Bool[Array, "d"]] = None,
    ) -> Float[Array, "*b"]:
        """Evaluate f at a single point x, vmap for more. Periodic dimensions wrap."""
        delta = x - self.x
        rbfs = jnp.where(
            False if periodic is None else periodic,
            periodic_rbf(self.l, delta),
            jnp.exp(-0.5 * delta**2 / self.l),
        )
        return jnp.sum(self.a * jnp.prod(rbfs, axis=-1), axis=-1)

    def pad_to(self, m: int) -> Self:
        """Pad with inert atoms: zero amplitude, period wide lengthscale, at the origin."""
        *b, m0, d = self.l.shape
        pad = (*b, m - m0, d)
        l = jnp.concat([self.l, jnp.full(pad, (2 * X_MAX) ** 2)], axis=-2)
        x = jnp.concat([self.x, jnp.zeros(pad)], axis=-2)
        a = jnp.concat([self.a, jnp.zeros((*b, m - m0))], axis=-1)
        return self._replace(l=l, x=x, a=a)


def periodic_rbf(
    l: Float[Array, "..."],
    x: Float[Array, "..."],
    expansion_terms: int = 4,
) -> Float[Array, "..."]:
    """Repeated rbf functions exp(-0.5 * x^2 / l) with period 2 X_MAX.
    Choses the evaluation path such that the error is below ~exp(-pi (terms + 1/2)^2).
    """
    # angle in [-pi, pi] and angular variance on the circle
    theta = jnp.pi * x / X_MAX
    theta = theta - 2 * jnp.pi * jnp.round(theta / (2 * jnp.pi))
    s = (jnp.pi / X_MAX) ** 2 * l
    n = jnp.arange(-expansion_terms, expansion_terms + 1, dtype=x.dtype)

    # narrow atoms: nearest images, error ~exp(-2 pi^2 (terms + 1/2)^2 / s)
    images = jnp.exp(-0.5 * (theta[..., None] + 2 * jnp.pi * n) ** 2 / s[..., None])
    images = images.sum(axis=-1)

    # wide atoms: Poisson dual Fourier series, error ~exp(-(terms + 1)^2 s / 2)
    harmonics = jnp.exp(-0.5 * n**2 * s[..., None]) * jnp.cos(n * theta[..., None])
    harmonics = jnp.sqrt(s / (2 * jnp.pi)) * harmonics.sum(axis=-1)

    return jnp.where(s < 2 * jnp.pi, images, harmonics)


@jax.jit
def inner(
    f1: RBFMixture["*b m1 d"],
    f2: RBFMixture["*b m2 d"],
    l0: Optional[Float[Array, "d"]] = None,
    periodic: Optional[Bool[Array, "d"]] = None,
) -> Float[Array, "*b"]:
    """Inner product in the rkhs of phi(.; l0) on [-X_MAX, X_MAX]^d, in L2 if l0 is None.
    RBF are analytic, so the rkhs inner product on the box is an isometry of the one on the line.
    See Steinwart, Hush, Scovel 2006, https://doi.org/10.1109/TIT.2006.881713
    The L2 inner product is scaled by 1 / sqrt(2 pi) per dimension.
    """
    # atom pairs on (*b, m, n, d)
    l1, x1 = f1.l[..., :, None, :], f1.x[..., :, None, :]
    l2, x2 = f2.l[..., None, :, :], f2.x[..., None, :, :]
    periodic = False if periodic is None else periodic

    if l0 is None:  # L2 inner product, scaled by 1 / sqrt(2 pi)^d
        # box dimensions keep only the mass of each atom product N(mu, s) inside [-X_MAX, X_MAX]
        ls = l1 + l2
        mu = (l2 * x1 + l1 * x2) / ls
        root_2s = jnp.sqrt(2.0 * l1 * l2 / ls)
        mass = 0.5 * (erf((X_MAX - mu) / root_2s) - erf((-X_MAX - mu) / root_2s))
        scale = jnp.sqrt(l1 * l2 / ls) * jnp.where(periodic, 1.0, mass)
        valid = True
    else:
        # deconvolved atom variances, patched where an atom does not fit the ambient rkhs
        v1 = l1 - l0 / 2
        v2 = l2 - l0 / 2
        valid = jnp.all((v1 > 0.0) & (v2 > 0.0), axis=(-3, -2, -1))
        ls = jnp.where(v1 > 0.0, v1, 1.0) + jnp.where(v2 > 0.0, v2, 1.0)
        scale = jnp.sqrt(l1 * l2 / (l0 * ls))

    # inner kernel matrix, infinite if any atom falls outside the ambient rkhs
    line = jnp.exp(-0.5 * (x1 - x2) ** 2 / ls)
    circle = periodic_rbf(ls, x1 - x2)
    K = jnp.prod(scale * jnp.where(periodic, circle, line), axis=-1)
    k = jnp.einsum("...ij,...i,...j->...", K, f1.a, f2.a)
    return jnp.where(valid, k, jnp.inf)


@jax.jit
def distance(
    f1: RBFMixture["*b m1 d"],
    f2: RBFMixture["*b m2 d"],
    l0: Optional[Float[Array, "d"]] = None,
    periodic: Optional[Bool[Array, "d"]] = None,
) -> Float[Array, "*b"]:
    """Distance between RBF mixtures in the rkhs of phi(.; l0), in L2 on the box if l0 is None."""
    norms = inner(f1, f1, l0, periodic) + inner(f2, f2, l0, periodic)
    d2 = norms - 2.0 * inner(f1, f2, l0, periodic)
    is_undefined = jnp.isinf(d2) | jnp.isnan(d2)  # inf - inf gives nan
    d2 = jnp.where(is_undefined, jnp.inf, d2)
    return utils.safe_sqrt(d2)
