import dataclasses
import functools
from typing import Callable, Literal, NamedTuple, Self
from jaxtyping import Array, Float, Scalar

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import equinox as eqx
from . import utils

InputType = Literal["box", "periodic"]
OutputType = Literal["tanh", "linear"]

# atoms of a box input may stick out of [0, 1] by this many of their own sigma
MAX_OVERHANG = 3.0

# periodic atoms are wrapped by summing this many images on each side of the nearest
# one: with squared lengthscales up to 2 (pair products) the first neglected image
# weighs exp(-6.5^2 / 4) ~ 1e-5
IMAGES = 6


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class Space:
    """Per dimension constraints of f: [0, 1]^d -> R^k, static under jit and tree maps.

    inputs: "box" lives on [0, 1] (atoms may overhang by MAX_OVERHANG sigma, the
    function only matters inside), "periodic" on the circle [0, 1) with wrapped atoms.
    outputs: "tanh" squashes into (-1, 1), "linear" leaves the mixture as is.
    An empty tuple means every dimension takes the first option.
    """

    inputs: tuple[InputType, ...] = ()
    outputs: tuple[OutputType, ...] = ()

    def __post_init__(self):
        for kind, allowed in [(self.inputs, InputType), (self.outputs, OutputType)]:
            if bad := set(kind) - set(allowed.__args__):
                raise ValueError(f"unknown constraint types {bad}, pick from {allowed.__args__}")

    @staticmethod
    def _mask(types: tuple[str, ...], n: int, value: str) -> np.ndarray:
        if types and len(types) != n:
            raise ValueError(f"space declares {len(types)} dimensions, the mixture has {n}")
        return np.array([t == value for t in types] if types else [False] * n)

    def periodic(self, d: int) -> np.ndarray:
        return self._mask(self.inputs, d, "periodic")

    def linear(self, k: int) -> np.ndarray:
        return self._mask(self.outputs, k, "linear")


def dim_product(
    box_fn: Callable[..., Array],
    line_fn: Callable[..., Array],
    periodic: np.ndarray,
    delta: Float[Array, "... d"],
    *params: Float[Array, "... d"],
) -> Float[Array, "..."]:
    """Product over dimensions of per dimension factors, wrapping the periodic ones.

    Box dimensions take box_fn(delta, *params). Periodic dimensions sum line_fn, the
    factor on the real line, over the images delta + n of the nearest offset.
    """
    shape = jnp.broadcast_shapes(delta.shape, *(p.shape for p in params))
    take = lambda z, i: jnp.broadcast_to(z, shape)[..., i]
    box, per = np.flatnonzero(~periodic), np.flatnonzero(periodic)
    out = jnp.prod(box_fn(take(delta, box), *(take(p, box) for p in params)), axis=-1)
    if len(per):
        dp = take(delta, per)
        dp = dp - jnp.round(dp)
        shifts = jnp.arange(-IMAGES, IMAGES + 1, dtype=dp.dtype)[:, None]
        expand = lambda z: take(z, per)[..., None, :]
        images = line_fn(dp[..., None, :] + shifts, *(expand(p) for p in params))
        out = out * jnp.prod(images.sum(axis=-2), axis=-1)
    return out


def gaussian(delta: Array, l: Array, *_) -> Array:
    return jnp.exp(-0.5 * delta**2 / l)


def centers_from_unit(u: Float[Array, "... d"], l: Float[Array, "... d"], space: Space):
    """Atom centers from unit coordinates: box atoms reach MAX_OVERHANG sigma outside."""
    reach = MAX_OVERHANG * jnp.sqrt(l)
    box = -reach + u * (1.0 + 2.0 * reach)
    return jnp.where(space.periodic(u.shape[-1]), u, box)


def centers_to_unit(x: Float[Array, "... d"], l: Float[Array, "... d"], space: Space):
    """Inverse of centers_from_unit, periodic centers taken modulo 1."""
    reach = MAX_OVERHANG * jnp.sqrt(l)
    box = (x + reach) / (1.0 + 2.0 * reach)
    return jnp.where(space.periodic(x.shape[-1]), x % 1.0, box)


class RBFMixture(NamedTuple):
    """Function h = squash(f) of an RBF mixture f = sum_i a_i prod_j phi(x_j - x_ij; l_ij).

    phi is a Gaussian on box inputs and a Gaussian wrapped around the circle on
    periodic ones, squash is tanh on bounded outputs and the identity on linear ones,
    as declared by space. The mixture f is the latent element of the rkhs that the GP
    compares and the acquisition optimizes, the target only ever sees h.
    """

    l: Float[Array, "... m d"]
    x: Float[Array, "... m d"]
    a: Float[Array, "... m"]
    space: Space = Space()

    @eqx.filter_jit
    def latent(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        """Evaluate the unsquashed mixture f at points x."""
        x = x[..., *(None,) * self.a.ndim, :]
        periodic = self.space.periodic(self.x.shape[-1])
        phi = dim_product(gaussian, gaussian, periodic, x - self.x, self.l)
        return jnp.sum(self.a * phi, axis=-1)

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        """Evaluate the squashed function h at points x, outputs on the last axis."""
        f = self.latent(x)
        return jnp.where(self.space.linear(self.a.shape[-2]), f, jnp.tanh(f))

    @staticmethod
    @eqx.filter_jit
    def from_lhs(
        key,
        shape: tuple[int, ...],
        l_range: tuple[Scalar, Scalar],
        a_range: tuple[Scalar, Scalar],
        space: Space = Space(),
    ) -> "RBFMixture":
        """Latin-hypercube mixtures with log-uniform squared lengthscales."""
        *batch, m, d = shape
        p = utils.latin_hypercube_sample(key, (*batch, m, 2 * d + 1))
        log_l, u, a = jnp.split(p, [d, 2 * d], axis=-1)
        l = jnp.exp(utils.rescale(log_l, jnp.log(l_range[0]), jnp.log(l_range[1])))
        x = centers_from_unit(u, l, space)
        a = utils.rescale(a, *a_range)
        return RBFMixture(l=l, x=x, a=a.squeeze(-1), space=space)

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


def _pairs(f1: RBFMixture, f2: RBFMixture):
    """Atom parameters broadcast to the common shape (..., m1, m2, d)."""
    l1, x1 = f1.l[..., :, None, :], f1.x[..., :, None, :]
    l2, x2 = f2.l[..., None, :, :], f2.x[..., None, :, :]
    return l1, x1, l2, x2, f1.space.periodic(f1.x.shape[-1])


@eqx.filter_jit
def rbf_inner(
    l0: Float[Array, "... d"],
    f1: RBFMixture,
    f2: RBFMixture,
) -> Float[Array, "..."]:
    """Inner product of RBF mixtures, seen in a wider RBF rkhs.

    On periodic inputs the ambient kernel is wrapped as well, and by Poisson
    summation the pair factor is the real line one summed over images.
    """
    l1, x1, l2, x2, periodic = _pairs(f1, f2)
    l0 = l0[..., None, None, :]

    # lengthscale factors, patched where the pair does not fit the ambient rkhs
    ls = l1 + l2 - l0
    valid = jnp.all(ls > 0.0, axis=(-3, -2, -1))
    ls = jnp.where(ls > 0.0, ls, 1.0)
    scale = jnp.sqrt(l1 * l2 / (l0 * ls))

    # inner kernel matrix, infinite if any pair falls outside the ambient rkhs
    factor = lambda delta, ls, scale: scale * gaussian(delta, ls)
    K = dim_product(factor, factor, periodic, x1 - x2, ls, scale)
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

    Per box dimension, the product of two Gaussian atoms is a Gaussian in t of
    variance s = l1 l2 / (l1 + l2) centered at mu = (x1 l2 + x2 l1) / (l1 + l2),
    times exp(-(x1 - x2)^2 / (2 (l1 + l2))), so it integrates to erf terms.
    With sigma, box dimensions are weighted by exp(-(t - center)^2 / (2 sigma^2)):
    one more Gaussian factor, so still erf terms, and sigma -> inf recovers the box.

    Per periodic dimension, the integral over the circle of two wrapped atoms is the
    real line integral sqrt(2 pi s) exp(-(x1 - x2)^2 / (2 (l1 + l2))) over images,
    unweighted since the circle has no edges.
    """
    l1, x1, l2, x2, periodic = _pairs(f1, f2)

    def box_fn(delta, l1, x1, l2, x2):
        ls = l1 + l2
        s = l1 * l2 / ls
        mu = (x1 * l2 + x2 * l1) / ls
        scale = gaussian(delta, ls)
        if sigma is not None:
            v = sigma**2
            scale = scale * gaussian(mu - center, s + v)
            s, mu = s * v / (s + v), (mu * v + center * s) / (s + v)
        w = jnp.sqrt(2.0 * s)
        return scale * 0.5 * jnp.sqrt(jnp.pi) * w * (erf((1.0 - mu) / w) + erf(mu / w))

    def line_fn(delta, l1, x1, l2, x2):
        ls = l1 + l2
        return jnp.sqrt(2.0 * jnp.pi * l1 * l2 / ls) * gaussian(delta, ls)

    K = dim_product(box_fn, line_fn, periodic, x1 - x2, l1, x1, l2, x2)
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
    """RMS distance between the squashed functions h over quadrature points.

    Unlike the closed-form distances it sees what the target sees, so functions that
    differ only where tanh saturates are close.
    """
    z = jnp.asarray(quadrature_points(f1.x.shape[-1]))
    d2 = jnp.mean(jnp.square(f1(z) - f2(z)), axis=0)
    d = jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0))
    return jnp.where(d2 > 0.0, d, 0.0)
