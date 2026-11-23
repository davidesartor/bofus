from types import GenericAlias
from typing import NamedTuple, Self
from jaxtyping import Array, Float, Key

import numpy as np
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import erf

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


def box_weight(x_range: XRange, d: int) -> RBFMixture["1 d"]:
    """One unit gaussian centred on the box with std its width."""
    lo, hi = (jnp.broadcast_to(jnp.asarray(z, float), (d,)) for z in x_range)
    return RBFMixture(((hi - lo) ** 2)[None], ((lo + hi) / 2)[None], jnp.ones(1))


@eqx.filter_jit
def inner(
    f1: RBFMixture["*b m1 d"],
    f2: RBFMixture["*b m2 d"],
    l0: Float[Array, "d"] | None,
    x_range: XRange,
    weight: RBFMixture["*b w d"],
    periodic: tuple[bool, ...] | None = None,
) -> Float[Array, "*b"]:
    """L2 inner product on the line of f1 w f2, scaled by 1 / sqrt(2 pi) per dimension.
    With l0 the atoms are deconvolved (variance l - l0 / 2), so a flat w gives the rkhs inner
    product of phi(.; l0), an isometry of the one on the box for analytic RBFs (Steinwart, Hush,
    Scovel 2006, https://doi.org/10.1109/TIT.2006.881713). w is flat where its l is inf and on
    periodic dims, whose periods x_range sets.
    """
    wraps = np.zeros(f1.x.shape[-1], bool) if periodic is None else np.asarray(periodic)

    if l0 is None:  # L2 inner product of the atoms themselves
        u1, u2, half_log_l0, valid = f1.l, f2.l, 0.0, True
    else:
        # deconvolved atoms, patched where an atom does not fit the ambient rkhs
        v1, v2 = f1.l - l0 / 2, f2.l - l0 / 2
        valid = jnp.all(v1 > 0.0, axis=(-2, -1)) & jnp.all(v2 > 0.0, axis=(-2, -1))
        u1, u2 = jnp.where(v1 > 0.0, v1, 1.0), jnp.where(v2 > 0.0, v2, 1.0)
        half_log_l0 = 0.5 * jnp.sum(jnp.log(l0))

    # per-atom part of the log scale 0.5 log(l1 l2 / (l0 E)), where u cancels; -0.5 log E is per dim
    half_log_l1 = 0.5 * jnp.sum(jnp.log(f1.l), axis=-1)[..., :, None]
    half_log_l2 = 0.5 * jnp.sum(jnp.log(f2.l), axis=-1)[..., None, :]
    log_norm = half_log_l1 + half_log_l2 - half_log_l0

    # weight precision r, zero where flat or periodic so those dims reduce to the plain rbf
    r = jnp.where(wraps, 0.0, 1.0 / weight.l)[..., None, None, :, :]
    z = weight.x[..., None, None, :, :]

    # each atom pair against each weight atom on (*b, m1, m2, w, d), one log and one divide each
    u1, x1 = u1[..., :, None, None, :], f1.x[..., :, None, None, :]
    u2, x2 = u2[..., None, :, None, :], f2.x[..., None, :, None, :]
    ls, dx = u1 + u2, x1 - x2
    E = ls + r * u1 * u2
    spread = r * (u1 * (x2 - z) ** 2 + u2 * (x1 - z) ** 2) + jnp.where(
        wraps, 0.0, dx**2
    )
    log_K = jnp.sum(-0.5 * jnp.log(E) - 0.5 * spread / E, axis=-1)
    if wraps.any():  # periodic dims go through log_rbf
        dx_wraps = jnp.where(wraps, dx[..., 0, :], 0.0)
        log_K = log_K + log_rbf(dx_wraps, ls[..., 0, :], x_range, periodic)[..., None]

    # atom-pair matrix summed over the weight atoms, inf if any atom falls outside the ambient rkhs
    K = jnp.sum(
        jnp.exp(log_norm[..., None] + log_K) * weight.a[..., None, None, :], axis=-1
    )
    k = jnp.einsum("...ij,...i,...j->...", K, f1.a, f2.a)
    return jnp.where(valid, k, jnp.inf)


@eqx.filter_jit
def squared_distance(
    f1: RBFMixture["*b m1 d"],
    f2: RBFMixture["*b m2 d"],
    l0: Float[Array, "d"] | None,
    x_range: XRange,
    weight: RBFMixture["*b w d"],
    periodic: tuple[bool, ...] | None = None,
) -> Float[Array, "*b"]:
    """Squared distance between RBF mixtures under the weighted inner product of inner.
    Broadcasts over *b, each norm computed at its own batch shape.
    """
    product = lambda a, b: inner(a, b, l0, x_range, weight, periodic)
    d2 = product(f1, f1) + product(f2, f2) - 2.0 * product(f1, f2)
    # inf - inf gives nan, cancellation can dip below zero
    return jnp.where(jnp.isfinite(d2), jnp.where(d2 > 0.0, d2, 0.0), jnp.inf)
