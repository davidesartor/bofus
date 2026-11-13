from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax.scipy.linalg import cho_solve, cholesky
from vlse import optim

from . import kernels, rkhs, utils


@jax.jit
def factorize(
    K: Float[Array, "n n"],
    y: Float[Array, "n"],
) -> tuple[Float[Array, "n n"], Float[Array, "n"], Float[Array, "n"], Scalar, Scalar]:
    """Cholesky factor and solves of K, with the trend b and scale nu profiled out.
    A nan observation marks padding, masked to an identity block of the factor.
    """
    mask = jnp.isfinite(y)
    y = jnp.where(mask, y, 0.0)
    K_chol = cholesky(utils.mask_covariance(K, mask), lower=True)

    # optimal trend b, weighting the observations by Ki_1 = K^-1 @ 1
    Ki_1 = cho_solve((K_chol, True), 1.0 * mask)
    b = jnp.average(y, weights=Ki_1)

    # optimal scale nu from the residual solve Ki_r = K^-1 @ (y - b)
    Ki_r = cho_solve((K_chol, True), y) - b * Ki_1
    nu = jnp.average((y - b) * Ki_r, weights=mask)
    return K_chol, Ki_1, Ki_r, b, nu


@jax.jit
def loglikelihood(
    K: Float[Array, "n n"],
    y: Float[Array, "n"],
) -> Scalar:
    """Marginal likelihood with the trend and scale concentrated out."""
    K_chol, _, _, _, nu = factorize(K, y)
    n = jnp.isfinite(y).sum()
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_chol)))
    return -0.5 * (n * jnp.log(nu) + logdetK)


@eqx.filter_jit
def pairwise_squared_distance(
    l0: Float[Array, "d"] | None,
    f1: rkhs.RBFMixture["n1 o m d"],
    f2: Optional[rkhs.RBFMixture["n2 o m d"]] = None,
    periodic: tuple[bool, ...] | None = None,
) -> Float[Array, "n1 n2 o"]:
    """Squared distances between two sets of functions, per output.
    Distances live in the rkhs of ambient lengthscale l0, in L2 on the box if l0 is None.
    """
    should_fill_diag = f2 is None
    f2 = f1 if f2 is None else f2

    f1 = jax.tree.map(lambda z: z[:, None], f1)
    f2 = jax.tree.map(lambda z: z[None, :], f2)
    d2 = rkhs.distance(f1, f2, l0, periodic) ** 2

    # floating point arithmetic is not guaranteed to cancel out
    if should_fill_diag:
        I = jnp.eye(len(d2), dtype=bool)[..., None]
        d2 = jnp.where(I, 0.0, d2)
    return d2


class GaussianProcess(NamedTuple):
    """Fitted GP over functions, compared through their ambient-space or L2 distances."""

    profile: kernels.Profile  # kernel profile, e.g. RBF or Matern
    l0: Float[Array, "d"] | None  # ambient lengthscale, None for L2
    periodic: tuple[bool, ...] | None  # dimensions that wrap, static; None for none

    # fitted parameters
    rho: Float[Array, "o"]  # weight output dimensions
    g: Scalar  # noise nugget
    nu: Scalar  # covariance scale
    b: Scalar  # mean trend

    # observations, nan values mark padded entries
    f: rkhs.RBFMixture["n o m d"]
    y: Float[Array, "n"]

    # cached for posterior evaluation, K is the covariance before scaling by nu
    K_chol: Float[Array, "n n"]
    Ki_1: Float[Array, "n"]  # K^-1 @ 1
    Ki_r: Float[Array, "n"]  # K^-1 @ (y - b)

    @staticmethod
    @eqx.filter_jit
    def fit(
        key: Key,
        f: rkhs.RBFMixture["n o m d"],
        y: Float[Array, "n"],
        *,
        profile: kernels.Profile = kernels.matern52,
        l0: Optional[Float[Array, "d"]] = None,
        periodic: tuple[bool, ...] | None = None,
        rho_range: Float[Array, "2"] = jnp.array([1e-2, 1e2]),
        nugget_range: Float[Array, "2"] = jnp.array([1e-4, 1e0]),
        restarts: int = 1,
    ) -> "GaussianProcess":
        """Maximum likelihood fit over (rho, g), best of several L-BFGS-B starts."""
        n, o, _, _ = f.l.shape

        # squared distances per output, constant throughout the optimisation
        mask = jnp.isfinite(y)
        d2 = pairwise_squared_distance(l0, f, periodic=periodic)

        # center rho to average inverse distances
        valid = (mask[:, None] & mask[None, :])[..., None] & jnp.isfinite(d2)
        valid = eqx.error_if(valid, ~jnp.any(valid, axis=(0, 1)), "no valid distances")
        rho_scale = 1 / jnp.mean(d2, axis=(0, 1), where=valid)

        def mle_loss(log_params: tuple[Float[Array, "o"], Scalar]) -> Scalar:
            log_rho, log_g = log_params
            rho, g = rho_scale * jnp.exp(log_rho), jnp.exp(log_g)
            K = profile(utils.safe_sqrt(d2 @ rho)) + g * jnp.eye(n)
            return -loglikelihood(K, y)

        # L-BFGS-B from uniform restarts over the log-space box
        (rho_lo, rho_hi), (g_lo, g_hi) = jnp.log(rho_range), jnp.log(nugget_range)
        key_rho, key_g = jr.split(key)
        log_rho = jr.uniform(key_rho, (restarts, o), minval=rho_lo, maxval=rho_hi)
        log_g = jr.uniform(key_g, (restarts,), minval=g_lo, maxval=g_hi)

        bounds = ((rho_lo, g_lo), (rho_hi, g_hi))
        solve = lambda start: optim.minimise(mle_loss, start, bounds=bounds)
        results = jax.vmap(solve)((log_rho, log_g))

        # extract the optimal parameters
        dead = ~jnp.any(jnp.isfinite(results.f))
        results = eqx.error_if(results, dead, "fit ended with non finite loss")
        best = jnp.nanargmin(results.f)
        log_rho, log_g = jax.tree.map(lambda z: z[best], results.x)
        rho, g = rho_scale * jnp.exp(log_rho), jnp.exp(log_g)

        # condition on the observations, caching the solves predict needs
        K = profile(utils.safe_sqrt(d2 @ rho)) + g * jnp.eye(n)
        K_chol, Ki_1, Ki_r, b, nu = factorize(K, y)
        return GaussianProcess(profile, l0, periodic, rho, g, nu, b, f, y, K_chol, Ki_1, Ki_r)

    @eqx.filter_jit
    def predict_joint(
        self, f: rkhs.RBFMixture["q o m d"]
    ) -> tuple[Float[Array, "q"], Float[Array, "q q"]]:
        """Posterior mean and covariance at f, including the estimated trend's uncertainty."""
        # cancellation only approximates the exact zeros of coincident functions
        d2 = lambda *fs: pairwise_squared_distance(self.l0, *fs, periodic=self.periodic)
        dist = lambda *fs: utils.safe_sqrt(d2(*fs) @ self.rho)
        Kqq = self.profile(dist(f))
        Knq = self.profile(dist(self.f, f))

        # a nan observation marks padding, zero its cross covariance
        Knq = Knq * jnp.isfinite(self.y)[:, None]

        # posterior mean and unit-scale covariance
        mean = self.b + Knq.T @ self.Ki_r
        cov = Kqq - Knq.T @ cho_solve((self.K_chol, True), Knq)

        # correction from estimating the trend on the same data
        Kbq = 1 - Knq.T @ self.Ki_1
        cov = cov + jnp.outer(Kbq, Kbq) / self.Ki_1.sum()
        return mean, self.nu * cov

    @eqx.filter_jit
    def predict_marginal(
        self, f: rkhs.RBFMixture["q o m d"]
    ) -> tuple[Float[Array, "q"], Float[Array, "q"]]:
        """Posterior mean and variance of each function on its own."""
        f = jax.tree.map(lambda z: z[:, None], f)
        mean, cov = jax.vmap(self.predict_joint)(f)
        return mean.squeeze(-1), cov.squeeze((-2, -1))
