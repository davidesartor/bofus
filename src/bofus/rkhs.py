from types import GenericAlias
from typing import NamedTuple, Self
from jaxtyping import Array, Float, Key

import numpy as np
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import erf

from . import utils

# (lo, hi) of the input box, of the squared lengthscales and of the amplitudes
XRange = tuple[float | Float[Array, "d"], float | Float[Array, "d"]]
LRange = tuple[float | Float[Array, "d"], float | Float[Array, "d"]]
ARange = tuple[float | Float[Array, ""], float | Float[Array, ""]]


class RBFMixture(NamedTuple):
    """RBF mixture f = sum_i a_i prod_j phi(x_j - x_ij; l_ij)."""

    l: Float[Array, "*b m d"]
    x: Float[Array, "*b m d"]
    a: Float[Array, "*b m"]

    @classmethod
    def __class_getitem__(cls, shape: str) -> GenericAlias:
        """Shape annotation RBFMixture["*b m d"], documentation only."""
        return GenericAlias(cls, shape)

    @eqx.filter_jit
    def __call__(
        self,
        x: Float[Array, "d"],
        x_range: XRange,
        periodic: tuple[bool, ...] | None = None,
    ) -> Float[Array, "*b"]:
        """Evaluate f at a single point x, vmap for more. Periodic dimensions wrap."""
        rbfs = jnp.exp(log_rbf(x - self.x, self.l, x_range, periodic))
        return jnp.sum(self.a * rbfs, axis=-1)

    @classmethod
    def sample(
        cls,
        key: Key,
        shape: tuple[int, ...],
        l_range: LRange,
        a_range: ARange,
        x_range: XRange,
        sigma0: float = 1.0,
        coverage: float = 1.0,
    ) -> Self:
        """Draw iid mixtures of shape (*b, m, d) from the prior, then clip l and a to their ranges.
        log l_j / w_j^2 ~ N(median, sigma0^2 / d) for half-widths w, the median solved so the atoms fill
        coverage of the box (m + 1) E[phi^2] = coverage. Amplitudes make the box mean of Var f(x) 1
        before clipping. Periodic dims use the line energy.
        """
        *b, m, d = shape
        lo, hi = x_range
        half_width = (jnp.asarray(hi) - lo) / 2
        key_l, key_x, key_a = jr.split(key, 3)
        sigma = sigma0 / np.sqrt(d)

        def log_energy(log_median: Float[Array, "s"]) -> Float[Array, "s"]:
            """log E[phi^2] over the box and the log-normal l, depends only on l / w^2 per dimension."""
            z = jnp.linspace(-6.0, 6.0, 512)
            r = jnp.exp(log_median[:, None] + sigma * z)
            # phi^2 = exp(-u^2 / r) averaged over the offset u of two uniform points, density (2 - |u|) / 4
            flat = jnp.sqrt(jnp.pi * r) * erf(2 / jnp.sqrt(r))
            ramp = 0.5 * r * jnp.expm1(-4 / r)
            energy = (flat + ramp) / 2
            return d * jnp.log(energy @ jax.nn.softmax(-0.5 * z**2))

        # coverage median interpolated over a wide range of candidate medians
        candidates = jnp.linspace(-20.0, 10.0, 512)
        log_target = jnp.log(coverage / (m + 1))
        log_median = jnp.interp(log_target, log_energy(candidates), candidates)

        # at the median m E[phi^2] = m coverage / (m + 1), the amplitude variance is its inverse
        l = half_width**2 * jnp.exp(log_median + sigma * jr.normal(key_l, shape))
        x = jr.uniform(key_x, shape, minval=lo, maxval=hi)
        a = jr.normal(key_a, (*b, m)) * jnp.sqrt((m + 1) / (m * coverage))
        return cls(l.clip(*l_range), x, a.clip(*a_range))

    def pad_to(self, key: Key, m: int, l_range: LRange, x_range: XRange) -> Self:
        """Pad to m atoms drawn from the prior with zero amplitude, so evaluations are unchanged."""
        *b, m0, d = self.l.shape
        pad = self.sample(key, (*b, m - m0, d), l_range, (0.0, 0.0), x_range)
        return jax.tree.map(lambda z, w: jnp.concat([z, w], axis=len(b)), self, pad)


def log_rbf(
    x: Float[Array, "... d"],
    l: Float[Array, "... d"],
    x_range: XRange,
    periodic: tuple[bool, ...] | None = None,
    expansion_terms: int = 4,
) -> Float[Array, "..."]:
    """log prod_j exp(-0.5 * x_j^2 / l_j), repeated with period hi_j - lo_j on periodic dims.
    Chooses the periodic evaluation path such that the error is below ~exp(-pi (terms + 1/2)^2).
    """
    if periodic is None or not any(periodic):
        return -0.5 * jnp.sum(x**2 / l, axis=-1)

    # only the dimensions that wrap pay for the periodic series
    wraps = np.asarray(periodic)
    line = -0.5 * jnp.sum(x[..., ~wraps] ** 2 / l[..., ~wraps], axis=-1)
    l, x = l[..., wraps], x[..., wraps]
    lo, hi = x_range
    period = jnp.broadcast_to(jnp.asarray(hi) - lo, wraps.shape)[wraps]

    # angle in [-pi, pi] and angular variance on the circle, each path sanitized for the other
    theta = 2 * jnp.pi * x / period
    theta = theta - 2 * jnp.pi * jnp.round(theta / (2 * jnp.pi))
    s = (2 * jnp.pi / period) ** 2 * l
    narrow = s < 2 * jnp.pi
    s_images, s_harmonics = jnp.where(narrow, s, 1.0), jnp.where(narrow, 2 * jnp.pi, s)
    n = jnp.arange(-expansion_terms, expansion_terms + 1, dtype=x.dtype)

    # narrow atoms: nearest images, error ~exp(-2 pi^2 (terms + 1/2)^2 / s)
    images = -0.5 * (theta[..., None] + 2 * jnp.pi * n) ** 2 / s_images[..., None]
    images = jax.nn.logsumexp(images, axis=-1)

    # wide atoms: Poisson dual Fourier series, error ~exp(-(terms + 1)^2 s / 2)
    decay = jnp.exp(-0.5 * n**2 * s_harmonics[..., None])
    harmonics = jnp.sum(decay * jnp.cos(n * theta[..., None]), axis=-1)
    harmonics = 0.5 * jnp.log(s_harmonics / (2 * jnp.pi)) + jnp.log(harmonics)

    circle = jnp.where(narrow, images, harmonics)
    return line + jnp.sum(circle, axis=-1)


@eqx.filter_jit
def inner(
    f1: RBFMixture["*b m1 d"],
    f2: RBFMixture["*b m2 d"],
    l0: Float[Array, "d"] | None,
    x_range: XRange,
    periodic: tuple[bool, ...] | None = None,
) -> Float[Array, "*b"]:
    """Inner product in the rkhs of phi(.; l0) on the box x_range, in L2 if l0 is None.
    RBF are analytic, so the rkhs inner product on the box is an isometry of the one on the line.
    See Steinwart, Hush, Scovel 2006, https://doi.org/10.1109/TIT.2006.881713
    The L2 inner product is scaled by 1 / sqrt(2 pi) per dimension.
    """
    TINY = utils.tiny(f1.x.dtype)
    lo, hi = x_range

    # atom pairs on (*b, m, n, d)
    l1, x1 = f1.l[..., :, None, :], f1.x[..., :, None, :]
    l2, x2 = f2.l[..., None, :, :], f2.x[..., None, :, :]
    wraps = np.zeros(f1.x.shape[-1], bool) if periodic is None else np.asarray(periodic)

    if l0 is None:  # L2 inner product, scaled by 1 / sqrt(2 pi)^d
        # box dimensions keep only the mass of each atom product N(mu, s) inside [lo, hi]
        ls = l1 + l2
        mu = (l2 * x1 + l1 * x2) / ls
        root_2s = jnp.sqrt(2.0 * l1 * l2 / ls)
        mass = 0.5 * (erf((hi - mu) / root_2s) - erf((lo - mu) / root_2s))
        mass = jnp.maximum(mass, TINY)  # finite log for atoms off the box
        log_scale = 0.5 * jnp.log(l1 * l2 / ls) + jnp.where(wraps, 0.0, jnp.log(mass))
        valid = True
    else:
        # deconvolved atom variances, patched where an atom does not fit the ambient rkhs
        v1 = l1 - l0 / 2
        v2 = l2 - l0 / 2
        valid = jnp.all((v1 > 0.0) & (v2 > 0.0), axis=(-3, -2, -1))
        ls = jnp.where(v1 > 0.0, v1, 1.0) + jnp.where(v2 > 0.0, v2, 1.0)
        log_scale = 0.5 * jnp.log(l1 * l2 / (l0 * ls))

    # inner kernel matrix, infinite if any atom falls outside the ambient rkhs
    K = jnp.exp(jnp.sum(log_scale, axis=-1) + log_rbf(x1 - x2, ls, x_range, periodic))
    k = jnp.einsum("...ij,...i,...j->...", K, f1.a, f2.a)
    return jnp.where(valid, k, jnp.inf)


@eqx.filter_jit
def distance(
    f1: RBFMixture["*b m1 d"],
    f2: RBFMixture["*b m2 d"],
    l0: Float[Array, "d"] | None,
    x_range: XRange,
    periodic: tuple[bool, ...] | None = None,
) -> Float[Array, "*b"]:
    """Distance between RBF mixtures in the rkhs of phi(.; l0), in L2 on the box if l0 is None."""
    norms = inner(f1, f1, l0, x_range, periodic) + inner(f2, f2, l0, x_range, periodic)
    d2 = norms - 2.0 * inner(f1, f2, l0, x_range, periodic)
    is_undefined = jnp.isinf(d2) | jnp.isnan(d2)  # inf - inf gives nan
    d2 = jnp.where(is_undefined, jnp.inf, d2)
    return utils.safe_sqrt(d2)
