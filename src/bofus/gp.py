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
    """Cholesky factor and solves of K, with the trend b and the REML scale nu.
    A nan observation marks padding, masked to an identity block of the factor.
    """
    # mask the covariance matrix and observations
    mask = jnp.isfinite(y)
    dof = jnp.maximum(mask.sum() - 1, 1)
    K = utils.mask_covariance(K, mask)
    y = jnp.where(mask, y, 0.0)

    # one solve for K^-1 @ 1 and K^-1 @ y, the trend b weights the observations by Ki_1
    K_chol = cholesky(K, lower=True)
    Ki_1, Ki_y = cho_solve((K_chol, True), jnp.stack([1.0 * mask, y], axis=-1)).T
    b = jnp.average(y, weights=Ki_1)

    # REML scale nu, one dof spent on b
    Ki_r = Ki_y - b * Ki_1
    nu = jnp.sum(jnp.where(mask, (y - b) * Ki_r, 0.0)) / dof

    # constant observations give nu = 0, floor it at their float resolution
    resolution = jnp.finfo(y.dtype).eps * jnp.max(jnp.abs(y))
    nu = jnp.maximum(nu, jnp.maximum(resolution**2, jnp.finfo(y.dtype).tiny))
    return K_chol, Ki_1, Ki_r, b, nu


@jax.jit
def loglikelihood(
    K: Float[Array, "n n"],
    y: Float[Array, "n"],
    nu: Scalar | None = None,
) -> Scalar:
    """REML likelihood: trend integrated out under a flat prior, scale nu concentrated unless given."""
    K_chol, Ki_1, _, _, nu_hat = factorize(K, y)
    dof = jnp.maximum(jnp.isfinite(y).sum() - 1, 1)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_chol))) + jnp.log(Ki_1.sum())
    if nu is None:
        return -0.5 * (dof * jnp.log(nu_hat) + logdetK)

    return -0.5 * (dof * jnp.log(nu) + logdetK + dof * nu_hat / nu)


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
    dof: Scalar | None  # student-t degrees of freedom
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
        y_std: Optional[Float[Array, "n"]] = None,
        profile: kernels.Profile = kernels.matern52,
        l0: Optional[Float[Array, "d"]] = None,
        periodic: tuple[bool, ...] | None = None,
        rho_range: Float[Array, "2"] = jnp.array([1e-2, 1e2]),
        nugget_range: Float[Array, "2"] = jnp.array([1e-4, 1e0]),
        nu_range: Float[Array, "2"] = jnp.array([1e-2, 1e2]),
        restarts: int = 1,
    ) -> "GaussianProcess":
        """MAP fit of (rho, g) on the REML likelihood, nu integrated out unless y_std is given."""
        n, o, _, _ = f.l.shape

        # squared distances per output, constant throughout the optimisation
        mask = jnp.isfinite(y)
        d2 = pairwise_squared_distance(l0, f, periodic=periodic)

        # center rho to average inverse distances
        valid = (mask[:, None] & mask[None, :])[..., None] & jnp.isfinite(d2)
        valid = eqx.error_if(valid, ~jnp.any(valid, axis=(0, 1)), "no valid distances")
        rho_scale = 1 / jnp.mean(d2, axis=(0, 1), where=valid)

        # known measurement variances, zeroed on padding so nan never reaches the gradients
        noise = None if y_std is None else jnp.where(mask, y_std, 0.0) ** 2

        def covariance(rho: Float[Array, "o"], g: Scalar, nu: Scalar | None) -> Float[Array, "n n"]:
            K = profile(utils.safe_sqrt(d2 @ rho)) + g * jnp.eye(n)
            return K if nu is None else K + jnp.diag(noise / nu)

        # center the nu search on the noiseless closed form at the box center
        g_center = jnp.sqrt(jnp.prod(nugget_range))
        nu_scale = factorize(covariance(rho_scale, g_center, None), y)[-1]

        def unpack(log_params: tuple[Float[Array, "..."], ...]) -> tuple:
            log_rho, log_g, *log_nu = log_params
            nu = nu_scale * jnp.exp(log_nu[0]) if log_nu else None
            return rho_scale * jnp.exp(log_rho), jnp.exp(log_g), nu

        # log-normal priors on (rho, g), box = ±3 std; nu stays flat in log, as in the noiseless branch
        ranges = (rho_range, nugget_range) + (() if y_std is None else (nu_range,))
        lo, hi = zip(*(jnp.log(r) for r in ranges))
        prior_mean = tuple((a + b) / 2 for a, b in zip(lo[:2], hi[:2]))
        prior_std = tuple((b - a) / 6 for a, b in zip(lo[:2], hi[:2]))

        def map_loss(log_params: tuple[Float[Array, "..."], ...]) -> Scalar:
            rho, g, nu = unpack(log_params)
            z = jax.tree.map(lambda u, m, s: (u - m) / s, log_params[:2], prior_mean, prior_std)
            neg_log_prior = 0.5 * sum(jnp.sum(zi**2) for zi in z)
            return -loglikelihood(covariance(rho, g, nu), y, nu) + neg_log_prior

        # L-BFGS-B from uniform restarts over the log-space box
        shapes = ((restarts, o), (restarts,), (restarts,))
        keys = jr.split(key, len(ranges))
        starts = tuple(
            jr.uniform(k, s, minval=a, maxval=b) for k, s, a, b in zip(keys, shapes, lo, hi)
        )

        solve = lambda start: optim.minimise(map_loss, start, bounds=(lo, hi))
        results = jax.vmap(solve)(starts)

        # extract the optimal parameters
        dead = ~jnp.any(jnp.isfinite(results.f))
        results = eqx.error_if(results, dead, "fit ended with non finite loss")
        best = jnp.nanargmin(results.f)
        rho, g, nu = unpack(jax.tree.map(lambda z: z[best], results.x))

        # condition on the observations, caching the solves predict needs
        K_chol, Ki_1, Ki_r, b, nu_hat = factorize(covariance(rho, g, nu), y)
        # integrating nu out leaves a student-t with n - 1 dof
        dof = None if nu is not None else jnp.maximum(mask.sum() - 1, 1).astype(y.dtype)
        nu = nu_hat if nu is None else nu
        return GaussianProcess(
            profile, l0, periodic, rho, g, nu, dof, b, f, y, K_chol, Ki_1, Ki_r
        )

    @eqx.filter_jit
    def predict_joint(
        self, f: rkhs.RBFMixture["q o m d"]
    ) -> tuple[Float[Array, "q"], Float[Array, "q q"]]:
        """Posterior location and scale at f, student-t with self.dof dof (gaussian if None)."""
        # cancellation only approximates the exact zeros of coincident functions
        dist = lambda *fs: utils.safe_sqrt(
            pairwise_squared_distance(self.l0, *fs, periodic=self.periodic) @ self.rho
        )
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
        """Posterior location and squared scale of each function on its own."""
        f = jax.tree.map(lambda z: z[:, None], f)
        mean, cov = jax.vmap(self.predict_joint)(f)
        return mean.squeeze(-1), cov.squeeze((-2, -1))
