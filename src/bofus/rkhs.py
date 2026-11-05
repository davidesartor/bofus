import functools
from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf
import equinox as eqx
from . import kernels, utils


class RBFMixture(NamedTuple):
    """Function h = tanh(f) of an RBF mixture f = sum_i a_i exp(-|x - x_i|^2 / 2 l_i).

    The mixture f is the latent element of the rkhs that the GP compares and the
    acquisition optimizes, the target only ever sees the squashed h in (-1, 1).
    """

    l: Float[Array, "... m d"]
    x: Float[Array, "... m d"]
    a: Float[Array, "... m"]

    @eqx.filter_jit
    def latent(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        """Evaluate the unsquashed mixture f at points x."""
        x = x[..., *(None,) * self.a.ndim, :]
        d = kernels.euclidean_distance(self.l, x, self.x)
        return jnp.sum(self.a * kernels.rbf(d), axis=-1)

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        """Evaluate the squashed function h = tanh(f) at points x."""
        return jnp.tanh(self.latent(x))

    @staticmethod
    @eqx.filter_jit
    def from_lhs(
        key,
        shape: tuple[int, ...],
        l_range: tuple[Scalar, Scalar],
        x_range: tuple[Scalar, Scalar],
        a_range: tuple[Scalar, Scalar],
    ) -> "RBFMixture":
        """Latin-hypercube mixtures with log-uniform squared lengthscales."""
        *batch, m, d = shape
        p = utils.latin_hypercube_sample(key, (*batch, m, 2 * d + 1))
        log_l, x, a = jnp.split(p, [d, 2 * d], axis=-1)
        log_l = utils.rescale(log_l, jnp.log(l_range[0]), jnp.log(l_range[1]))
        x = utils.rescale(x, *x_range)
        a = utils.rescale(a, *a_range)
        return RBFMixture(l=jnp.exp(log_l), x=x, a=a.squeeze(-1))

    def pad_to(self, m: int) -> Self:
        """Pad the mixture to a larger number of basis points."""
        *b, m0, d = self.l.shape
        l = jnp.ones((*b, m, d)).at[..., :m0, :].set(self.l)
        x = jnp.zeros((*b, m, d)).at[..., :m0, :].set(self.x)
        a = jnp.zeros((*b, m)).at[..., :m0].set(self.a)
        return self._replace(l=l, x=x, a=a)

    def split(self) -> Self:
        """Split every atom in two with half amplitude, leaving the function unchanged."""
        l = jnp.concatenate([self.l, self.l], axis=-2)
        x = jnp.concatenate([self.x, self.x], axis=-2)
        a = jnp.concatenate([self.a / 2, self.a / 2], axis=-1)
        return self._replace(l=l, x=x, a=a)


@eqx.filter_jit
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
    K = scale * kernels.rbf(kernels.euclidean_distance(ls, x1, x2))
    k = jnp.einsum("...ij,...i,...j->...", K, f1.a, f2.a)
    return jnp.where(valid, k, jnp.inf)


# width of the gaussian measure: a standard normal observation squashed by a sigmoid,
# as the closed loop control targets do, has a std of about 0.21 around 0.5
GAUSS_CENTER, GAUSS_SIGMA = 0.5, 0.25


@eqx.filter_jit
def l2_inner(
    f1: RBFMixture,
    f2: RBFMixture,
    sigma: float | None = None,
    center: float = GAUSS_CENTER,
) -> Float[Array, "..."]:
    """L2 inner product of RBF mixtures over the unit box [0, 1]^d, in closed form.

    Per dimension, the product of two Gaussian atoms is a Gaussian in t of
    variance s = l1 l2 / (l1 + l2) centered at mu = (x1 l2 + x2 l1) / (l1 + l2),
    times exp(-(x1 - x2)^2 / (2 (l1 + l2))), so it integrates to erf terms.

    With sigma, the box is weighted by exp(-(t - center)^2 / (2 sigma^2)): one more
    Gaussian factor, so still erf terms, and sigma -> inf recovers the plain box.
    """
    # broadcast to common shape (..., m1, m2, d)
    l1, x1 = f1.l[..., :, None, :], f1.x[..., :, None, :]
    l2, x2 = f2.l[..., None, :, :], f2.x[..., None, :, :]

    ls = l1 + l2
    s = l1 * l2 / ls
    mu = (x1 * l2 + x2 * l1) / ls
    scale = jnp.exp(-0.5 * (x1 - x2) ** 2 / ls)
    if sigma is not None:
        v = sigma**2
        scale = scale * jnp.exp(-0.5 * (mu - center) ** 2 / (s + v))
        s, mu = s * v / (s + v), (mu * v + center * s) / (s + v)
    w = jnp.sqrt(2.0 * s)
    box = 0.5 * jnp.sqrt(jnp.pi) * w * (erf((1.0 - mu) / w) + erf(mu / w))
    K = jnp.prod(scale * box, axis=-1)
    return jnp.einsum("...ij,...i,...j->...", K, f1.a, f2.a)


@functools.cache
def quadrature_points(d: int, n: int = 256, seed: int = 0) -> np.ndarray:
    """Fixed latin hypercube points on the unit box, a constant under jit."""
    rng = np.random.default_rng(seed)
    perm = np.stack([rng.permutation(n) for _ in range(d)], axis=-1)
    return ((perm + rng.uniform(size=(n, d))) / n).astype(np.float32)


def _distance(inner, f1: RBFMixture, f2: RBFMixture) -> Float[Array, "..."]:
    """Norm of the difference, expanded through an inner product."""
    d2 = inner(f1, f1) + inner(f2, f2) - 2.0 * inner(f1, f2)
    # fix nan and inf gradients returning the subgradient instead
    d2 = jnp.nan_to_num(d2, nan=jnp.inf, posinf=jnp.inf)
    d = jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0))
    return jnp.where(d2 > 0.0, d, 0.0)


@eqx.filter_jit
def rbf_distance(
    l0: Float[Array, "... d"],
    f1: RBFMixture,
    f2: RBFMixture,
) -> Float[Array, "..."]:
    """Distance between RBF mixtures, seen in a wider RBF rkhs."""
    return _distance(lambda u, v: rbf_inner(l0, u, v), f1, f2)


@eqx.filter_jit
def l2_distance(f1: RBFMixture, f2: RBFMixture) -> Float[Array, "..."]:
    """L2 distance between the latent mixtures over the unit box [0, 1]^d."""
    return _distance(l2_inner, f1, f2)


@eqx.filter_jit
def gauss_distance(f1: RBFMixture, f2: RBFMixture) -> Float[Array, "..."]:
    """L2 distance between the latent mixtures, Gaussian weighted around the box center."""
    return _distance(lambda u, v: l2_inner(u, v, sigma=GAUSS_SIGMA), f1, f2)


@eqx.filter_jit
def quad_distance(f1: RBFMixture, f2: RBFMixture) -> Float[Array, "..."]:
    """RMS distance between the squashed functions tanh(f) over quadrature points.

    Unlike the closed-form distances it sees what the target sees, so functions that
    differ only where tanh saturates are close.
    """
    z = jnp.asarray(quadrature_points(f1.x.shape[-1]))
    d2 = jnp.mean(jnp.square(f1(z) - f2(z)), axis=0)
    d = jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0))
    return jnp.where(d2 > 0.0, d, 0.0)
