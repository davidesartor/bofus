from typing import Callable, Literal, NamedTuple, Self
from jaxtyping import Array, Float, Scalar

import jax.numpy as jnp
from jax.scipy.special import erf
import equinox as eqx
from . import kernels, utils


class RBFMixture(NamedTuple):
    l: Float[Array, "... m d"]
    x: Float[Array, "... m d"]
    a: Float[Array, "... m"]

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        """Evaluate the function at points x."""
        x = x[..., *(None,) * self.a.ndim, :]
        d = kernels.euclidean_distance(self.l, x, self.x)
        y = jnp.sum(self.a * kernels.rbf(d), axis=-1)
        return y

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


Squash = Literal["none", "tanh"]


def squashed(f: RBFMixture, squash: Squash) -> Callable[[Float[Array, "d"]], Float[Array, "k"]]:
    """The function the target sees, h = squash(f), with the GP still modelling f.

    "none" hands f over as is (targets clip to their nominal [-1, 1] range), "tanh"
    maps it smoothly into (-1, 1) so every candidate is feasible.
    """
    if squash == "none":
        return f
    if squash == "tanh":
        return lambda x: jnp.tanh(f(x))
    raise ValueError(f"unknown squash {squash!r}, pick from {Squash.__args__}")


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


@eqx.filter_jit
def l2_inner(f1: RBFMixture, f2: RBFMixture) -> Float[Array, "..."]:
    """L2 inner product of RBF mixtures over the unit box [0, 1]^d, in closed form.

    Per dimension, the product of two Gaussian atoms is a Gaussian in t of
    variance s = l1 l2 / (l1 + l2) centered at mu = (x1 l2 + x2 l1) / (l1 + l2),
    times exp(-(x1 - x2)^2 / (2 (l1 + l2))), so it integrates to erf terms.
    """
    # broadcast to common shape (..., m1, m2, d)
    l1, x1 = f1.l[..., :, None, :], f1.x[..., :, None, :]
    l2, x2 = f2.l[..., None, :, :], f2.x[..., None, :, :]

    ls = l1 + l2
    s = l1 * l2 / ls
    mu = (x1 * l2 + x2 * l1) / ls
    w = jnp.sqrt(2.0 * s)
    box = 0.5 * jnp.sqrt(jnp.pi) * w * (erf((1.0 - mu) / w) + erf(mu / w))
    K = jnp.prod(jnp.exp(-0.5 * (x1 - x2) ** 2 / ls) * box, axis=-1)
    return jnp.einsum("...ij,...i,...j->...", K, f1.a, f2.a)


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
    """L2 distance between RBF mixtures over the unit box [0, 1]^d."""
    return _distance(l2_inner, f1, f2)
