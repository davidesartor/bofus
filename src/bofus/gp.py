from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax.scipy.linalg import cho_factor, cho_solve
from einops import unpack
from vlse import optim

from . import kernels, rkhs, utils


@jax.jit
def trend_and_scale(
    Koo_chol: Float[Array, "o o"],
    y: Float[Array, "o"],
) -> tuple[Scalar, Scalar]:
    """Profile estimates of the trend b and scale nu, concentrated out of the likelihood."""

    # a nan observation marks padding, mask it out everywhere
    mask = jnp.isfinite(y)
    Koo_chol = utils.mask_covariance(Koo_chol, mask)
    y = jnp.where(mask, y, 0.0)

    # Ki_1 = K^-1 @ 1 and Ki_y = K^-1 @ y
    Ki_1 = cho_solve((Koo_chol, False), 1.0 * mask)
    Ki_y = cho_solve((Koo_chol, False), y)

    # optimal trend b and scale nu
    b = jnp.average(y, weights=Ki_1)
    nu = jnp.average((y - b) * (Ki_y - Ki_1 * b), weights=mask)
    return b, nu


@jax.jit
def loglikelihood(
    Koo: Float[Array, "o o"],
    y: Float[Array, "o"],
) -> Scalar:
    """Marginal likelihood with the trend and scale concentrated out."""

    # a nan observation marks padding, mask it out everywhere
    mask = jnp.isfinite(y)
    Koo = utils.mask_covariance(Koo, mask)
    y = jnp.where(mask, y, 0.0)
    n = mask.sum()

    # factorize the covariance and compute its log determinant
    Koo_chol, _ = cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(Koo_chol)))

    # profile the trend and scale out of the likelihood
    _, nu = trend_and_scale(Koo_chol, y)
    return -0.5 * (n * jnp.log(nu) + logdetK)


@eqx.filter_jit
def pairwise_distance(
    l0: Float[Array, "k d"],
    rho: Float[Array, "k"],
    f1: rkhs.RBFMixture,  # assumed to be (n1, k, m, d)
    f2: Optional[rkhs.RBFMixture] = None,  # assumed to be (n2, k, m, d)
) -> Float[Array, "n1 n2"]:
    """Ambient-space distance matrix between two sets of functions."""
    should_fill_diag = f2 is None
    f2 = f1 if f2 is None else f2

    # get the weighted distance
    f1 = jax.tree.map(lambda z: z[:, None], f1)
    f2 = jax.tree.map(lambda z: z[None, :], f2)
    d2 = jnp.sum(rho * rkhs.rbf_distance(l0, f1, f2) ** 2, axis=-1)

    # floating point arithmetic is not guaranteed to cancel out
    if should_fill_diag:
        d2 = jnp.fill_diagonal(d2, 0.0, inplace=False)

    # fix nan and inf gradients returning the subgradient instead
    d2 = jnp.nan_to_num(d2, nan=jnp.inf, posinf=jnp.inf)
    d = jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0))
    return jnp.where(d2 > 0.0, d, 0.0)


class GaussianProcess(NamedTuple):
    """Fitted GP over functions, compared through their ambient-space distances."""

    profile: kernels.Profile

    # fitted parameters
    l0: Float[Array, "k d"]  # ambient lengthscale
    rho: Float[Array, "k"]  # weight output dimensions
    g: Scalar  # noise nugget
    nu: Scalar  # covariance scale
    b: Scalar  # mean trend

    # observations, nan values mark padded entries
    x: rkhs.RBFMixture
    y: Float[Array, "o"]
    Koo_chol: Float[Array, "o o"]  # cached for posterior evaluation

    @staticmethod
    @eqx.filter_jit
    def fit(
        f: rkhs.RBFMixture,  # assumed to be (o, k, m, d)
        y: Float[Array, "o"],
        *,
        key: Key = jr.key(42),
        n_starts: int = 8,
        profile: kernels.Profile = kernels.matern52,
        l0_range: tuple[Scalar, Scalar] = (jnp.array(1e-2), jnp.array(1e0)),
        rho_range: tuple[Scalar, Scalar] = (jnp.array(1e-2), jnp.array(1e2)),
        nugget_range: tuple[Scalar, Scalar] = (jnp.array(1e-4), jnp.array(1e0)),
    ):

        # clip l0 below the observed lengthscales so every pair fits the ambient rkhs
        l0_range = (l0_range[0], l0_range[1].clip(max=f.l.min()))
        log_l0_range = (jnp.log(l0_range[0]), jnp.log(l0_range[1]))
        log_rho_range = (jnp.log(rho_range[0]), jnp.log(rho_range[1]))
        log_g_range = (jnp.log(nugget_range[0]), jnp.log(nugget_range[1]))
        bounds = tuple(zip(log_l0_range, log_rho_range, log_g_range))

        # latin hypercube starts over the log-space box
        _, k, _, d = f.l.shape
        p = utils.latin_hypercube_sample(key, (n_starts, k * d + k + 1))
        log_l0, log_rho, log_g = unpack(p, [[k, d], [k], []], "n *")
        log_l0 = utils.rescale(log_l0, *log_l0_range)
        log_rho = utils.rescale(log_rho, *log_rho_range)
        log_g = utils.rescale(log_g, *log_g_range)

        # multistart L-BFGS-B, keep the best run
        def mle_loss(log_params) -> Scalar:
            l0, rho, g = jax.tree.map(jnp.exp, log_params)
            Koo = profile(pairwise_distance(l0, rho, f)) + g * jnp.eye(len(y))
            return -loglikelihood(Koo, y)

        solve = lambda log_params: optim.minimise(mle_loss, log_params, bounds=bounds)
        results = jax.vmap(solve)((log_l0, log_rho, log_g))

        # extract the optimal parameters
        all_dead = ~jnp.any(jnp.isfinite(results.f))
        results = eqx.error_if(results, all_dead, "all fits ended with non finite loss")
        best = jnp.nanargmin(results.f)
        l0, rho, g = jax.tree.map(lambda z: jnp.exp(z[best]), results.x)

        # infer the remaining parameters
        Koo = profile(pairwise_distance(l0, rho, f)) + g * jnp.eye(len(y))
        mask = jnp.isfinite(y)
        Koo = utils.mask_covariance(Koo, mask)
        b, nu = trend_and_scale(cho_factor(Koo)[0], y)
        Koo = utils.mask_covariance(nu * Koo, mask)
        Koo_chol, _ = cho_factor(Koo)
        return GaussianProcess(profile, l0, rho, g, nu, b, f, y, Koo_chol)

    @eqx.filter_jit
    def predict(
        self, f: rkhs.RBFMixture  # assumed to be (q, k, m, d)
    ) -> tuple[Float[Array, "q"], Float[Array, "q q"]]:
        # cancellation only approximates the exact zeros of coincident functions
        Kxx = self.nu * self.profile(pairwise_distance(self.l0, self.rho, f))
        Kox = self.nu * self.profile(pairwise_distance(self.l0, self.rho, self.x, f))

        # a nan observation marks padding, mask it out everywhere
        mask = jnp.isfinite(self.y)
        Koo_chol = utils.mask_covariance(self.Koo_chol, mask)
        Kox = Kox * mask[:, None]
        y = jnp.where(mask, self.y, 0.0)

        # posterior mean and covariance
        gain = cho_solve((Koo_chol, False), Kox).T
        mean = self.b + gain @ (y - self.b)
        cov = Kxx - gain @ Kox

        # correction from estimating the trend on the same data
        Kbx = 1 - gain @ mask
        Koo_inv_sum = mask @ cho_solve((Koo_chol, False), 1.0 * mask)
        cov = cov + jnp.outer(Kbx, Kbx) / Koo_inv_sum
        return mean, cov
