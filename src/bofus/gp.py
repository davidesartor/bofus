from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar
from functools import partial
import warnings

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import scipy as sp
from einops import rearrange, reduce

from . import kernels, rkhs

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


FALLBACK_LENGTHSCALE_RANGE = (EPS, 10.0)
DEFAULT_NUGGET_RANGE = (EPS, 1e2)


def padded_to(size: int, target: int | None = None) -> int:
    """Padding target for an axis: an explicit size, else the next power of two."""
    # doubling keeps the number of distinct shapes logarithmic without a known final size
    return target if target is not None else 1 << max(size - 1, 0).bit_length()


def inverse_profile(profile: kernels.Profile, correlation: float) -> float:
    """Squared distance at which the profile decays to the given correlation."""
    # every profile is monotone decreasing, so a bracketed root is unique
    fun = lambda d2: float(profile(jnp.asarray(d2))) - correlation
    return float(sp.optimize.brentq(fun, 0.0, 1e4, xtol=EPS))


def auto_lengthscale_range(
    profile: kernels.Profile,
    d2: Float[Array, "n n"],
    min_correlation: float = 0.01,
    max_correlation: float = 0.5,
    quantile: float = 0.05,
) -> tuple[float, float] | None:
    """hetGP-style lengthscale bounds from distance quantiles, for a profile(d2 / rho) kernel.

    None when no pair of distinct points is available to estimate the quantiles from.
    """
    pairs = d2[jnp.tril(d2, k=-1) > 0]
    if pairs.size == 0:
        return None

    # close points must be free to decorrelate, distant ones free to stay correlated
    low, high = jnp.quantile(pairs, jnp.array([quantile, 1.0 - quantile]))
    lower = max(float(low) / inverse_profile(profile, min_correlation), EPS)
    upper = max(float(high) / inverse_profile(profile, max_correlation), lower)
    return (lower, upper)


class Module(eqx.Module):
    def _replace(self, **kwargs) -> Self:
        where = lambda m: tuple(getattr(m, k) for k in kwargs.keys())
        return eqx.tree_at(where, self, kwargs.values(), is_leaf=lambda x: x is None)


class Gaussian(NamedTuple):
    mean: Float[Array, "n"]
    cov: Float[Array, "n n"]


@jax.jit
def gp_posterior(
    Kxx: Float[Array, "m m"],
    Kox: Float[Array, "n m"],
    Koo: Float[Array, "n n"] | None,
    observed_ys: Float[Array, "n"],
    b: Scalar,
    Koo_lu: tuple | None = None,
    Koo_inv_sum: Scalar | None = None,
) -> Gaussian:
    # Koo_lu and Koo_inv_sum depend only on the fitted model, so they can be cached
    if Koo_lu is None:
        Koo_lu = jsp.linalg.lu_factor(Koo)
    if Koo_inv_sum is None:
        Koo_inv_sum = jnp.linalg.inv(Koo).sum()

    # posterior mean and covariance
    gain = jsp.linalg.lu_solve(Koo_lu, Kox).T
    mean = b + gain @ (observed_ys - b)
    cov = Kxx - gain @ Kox

    # Add correction based on the trend estimation correlation
    Kbx = jnp.ones((1, len(observed_ys))) @ gain.T
    cov = cov + (1 - Kbx).T @ (1 - Kbx) / Koo_inv_sum
    return Gaussian(mean=mean, cov=cov)


@jax.jit
def loglikelihood(
    Koo: Float[Array, "n n"],
    ys: Float[Array, "n"],
    mask: Float[Array, "n"] | None = None,
) -> tuple[Scalar, Scalar, Scalar]:
    # a zero mask drops padded observations, so n can stay constant as data accumulates
    mask = jnp.ones_like(ys) if mask is None else mask
    n = mask.sum()

    # cholesky of K and compute logdet
    K_sqrt, is_lower = jsp.linalg.cho_factor(Koo)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))

    # compute Ki_1=(K^-1 @ 1) and Ki_y=(K^-1 @ y)
    Ki_1, Ki_y = jsp.linalg.cho_solve(
        c_and_lower=(K_sqrt, is_lower),
        b=jnp.stack([mask, ys], 1),
    ).T

    # compute optimal trend b and scale nu
    b = (Ki_1 * ys).sum() / Ki_1.sum()
    nu = jnp.dot((ys - b) * mask / n, (Ki_y - Ki_1 * b))

    # likelihood when marginalizing over trend and variance
    loglik = -0.5 * (n * jnp.log(nu) + logdetK)
    return (loglik, b, nu)


@eqx.filter_jit
def masked_kernel_matrix(
    metric: kernels.Metric,
    profile: kernels.Profile,
    rho: Float[Array, "D"],
    xs: Float[Array, "N D"],
    g: Scalar,
    mask: Float[Array, "N"],
) -> Float[Array, "N N"]:
    """Observation covariance with padded entries replaced by an inert identity block."""
    Koo = profile(metric(rho, xs, xs)) * mask[:, None] * mask[None, :]
    return Koo + jnp.diag(g * mask + (1 - mask))


@eqx.filter_jit
@partial(jax.value_and_grad, argnums=0)
def vector_mle_loss(
    params: Float[Array, "D+1"],
    metric: kernels.Metric,
    profile: kernels.Profile,
    xs: Float[Array, "N D"],
    ys: Float[Array, "N"],
    mask: Float[Array, "N"],
) -> Scalar:
    """Negative marginal likelihood of the lengthscales and nugget, at a fixed padded size."""
    rho, g = params[:-1], params[-1]
    Koo = masked_kernel_matrix(metric, profile, rho, xs, g, mask)
    return -loglikelihood(Koo, ys, mask)[0]


class GaussianProcess(Module):
    # kernel definition
    metric: kernels.Metric = kernels.Euclidean()
    profile: kernels.Profile = kernels.SquaredExponential()

    # model parameters
    rho: Float[Array, "d"] = eqx.field(default=None)
    g: Scalar = eqx.field(default=None)
    nu: Scalar = eqx.field(default=None)
    b: Scalar = eqx.field(default=None)

    # observed data, plus a copy padded to a fixed size and dimension for predict
    observed_xs: Float[Array, "N D"] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)
    padded_ys: Float[Array, "N"] = eqx.field(default=None)
    mask: Float[Array, "N"] = eqx.field(default=None)

    # cached covariance matrix of the observed ys, and factorizations of nu * Koo
    Koo: Float[Array, "n n"] = eqx.field(default=None)
    Koo_lu: tuple = eqx.field(default=None)
    Koo_inv_sum: Scalar = eqx.field(default=None)

    @eqx.filter_jit
    def kernel(
        self,
        rho: Float[Array, "d"],
        xs1: Float[Array, "m d"],
        xs2: Float[Array, "n d"],
    ) -> Float[Array, "m n"]:
        return self.profile(self.metric(rho, xs1, xs2))

    @eqx.filter_jit
    def predict(self, xs: Float[Array, "m d"]) -> Gaussian:
        # queries can be narrower than the padded observations, so widen them to match
        D = self.observed_xs.shape[-1]
        xs = jnp.zeros((len(xs), D)).at[:, : xs.shape[-1]].set(xs)

        # compute covariance matrices
        Kxx = self.nu * self.kernel(self.rho, xs, xs)
        Kox = self.nu * self.kernel(self.rho, self.observed_xs, xs)

        # padded observations are not inert in the kernel, so mask them out
        Kox = Kox * self.mask[:, None]
        return gp_posterior(
            Kxx,
            Kox,
            None,
            self.padded_ys,
            self.b,
            Koo_lu=self.Koo_lu,
            Koo_inv_sum=self.Koo_inv_sum,
        )

    def fit(
        self,
        xs: Float[Array, "n d"],
        ys: Float[Array, "n"],
        *,
        warmstart: bool = False,
        lengthscale_range: tuple[float, float] | None = None,
        nugget_range: tuple[float, float] = DEFAULT_NUGGET_RANGE,
        max_iterations: int = 100,
        ftol: float = EPS,
        gtol: float = 0.0,
        padded_size: int | None = None,
        padded_dim: int | None = None,
    ) -> Self:
        # pad to a stable size, so the mle loss retraces only when the padding grows
        n, d = xs.shape
        N, D = padded_to(n, padded_size), padded_to(d, padded_dim)
        padded_xs = jnp.zeros((N, D)).at[:n, :d].set(xs)
        mask = jnp.zeros(N).at[:n].set(1.0)
        padded_ys = jnp.zeros(N).at[:n].set(ys)

        def verbose_loss(params: Float[Array, "D+1"]):
            val, grad = vector_mle_loss(
                params, self.metric, self.profile, padded_xs, padded_ys, mask
            )
            if jnp.isnan(val) or jnp.isnan(grad).any():
                warnings.warn(f"NaN detected in loss or gradient: {params}")
            return val, grad

        # bounds are estimated on the unit cube, then stretched back to each input span
        input_spans = xs.max(0) - xs.min(0)
        spans = jnp.ones(D).at[:d].set(jnp.where(input_spans > 0, input_spans, 1.0))
        rescaled_xs = (xs - xs.min(0)) / spans[:d]
        auto_range = (
            auto_lengthscale_range(
                self.profile, kernels.Euclidean()(1.0, rescaled_xs, rescaled_xs)
            )
            if lengthscale_range is None
            else None
        )

        # initialization
        nugget = min(0.1, nugget_range[1])
        if auto_range is not None:
            # a lengthscale divides a distance here, so undo the profile's squaring
            lower, upper = spans * auto_range[0] ** 0.5, spans * auto_range[1] ** 0.5
            lengthscale = jnp.sqrt(lower * upper)
        else:
            lengthscale_range = lengthscale_range or FALLBACK_LENGTHSCALE_RANGE
            lower = jnp.full(D, lengthscale_range[0])
            upper = jnp.full(D, lengthscale_range[1])
            lengthscale = 0.9 * lower + 0.1 * upper
        if warmstart:
            nugget = self.g if self.g is not None else nugget
            lengthscale = self.rho if self.rho is not None else lengthscale
        init_params = jnp.concat(
            [jnp.broadcast_to(lengthscale, (D,)), jnp.array([nugget])]
        )

        # run optimization
        result = sp.optimize.minimize(
            fun=verbose_loss,
            x0=init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=[*zip(lower.tolist(), upper.tolist()), nugget_range],
            options=dict(maxiter=max_iterations, ftol=ftol, gtol=gtol),
        )

        # extract the optimal parameters and infer the rest, still at the padded size
        rho = jnp.array(result.x[:-1])
        g = jnp.array(result.x[-1])
        padded_Koo = masked_kernel_matrix(
            self.metric, self.profile, rho, padded_xs, g, mask
        )
        llk, b, nu = loglikelihood(padded_Koo, padded_ys, mask)

        # an identity block keeps the padding out of the real block's factorization
        scaled_Koo = nu * padded_Koo * mask[:, None] * mask[None, :]
        scaled_Koo = scaled_Koo + jnp.diag(1 - mask)

        # factorize once here instead of on every posterior evaluation
        Koo_lu = jsp.linalg.lu_factor(scaled_Koo)
        Koo_inv_sum = (jnp.linalg.inv(scaled_Koo) * mask[:, None] * mask[None, :]).sum()

        # return a new instance with the fitted parameters and observed data
        return self._replace(
            rho=rho,
            g=g,
            nu=nu,
            b=b,
            Koo=padded_Koo[:n, :n],
            Koo_lu=Koo_lu,
            Koo_inv_sum=Koo_inv_sum,
            observed_xs=padded_xs,
            observed_ys=ys,
            padded_ys=padded_ys,
            mask=mask,
        )


class Basis(NamedTuple):
    """Basis points and coefficients of a batch of rkhs.Function, padded to a common size."""

    kernel: rkhs.RKHS
    x: Float[Array, "n k d"]
    a: Float[Array, "n k m"]

    @classmethod
    def stack(
        cls, fs: list[rkhs.Function], n: int | None = None, k: int | None = None
    ) -> Self:
        """Stack functions, optionally padding to n functions of k basis points each."""
        # zero coefficients make the padding inert in every inner product
        k = padded_to(max(len(f.x) for f in fs), k)
        pad = lambda z: jnp.concat([z, jnp.zeros((k - len(z), *z.shape[1:]))])
        x = jnp.stack([pad(f.x) for f in fs])
        a = jnp.stack([pad(f.a) for f in fs])

        # padding functions keep the shapes constant as observations accumulate
        if n is not None and n > len(fs):
            x = jnp.concat([x, jnp.zeros((n - len(fs), *x.shape[1:]))])
            a = jnp.concat([a, jnp.zeros((n - len(fs), *a.shape[1:]))])
        return cls(kernel=fs[0].kernel, x=x, a=a)


@eqx.filter_jit
def rkhs_inner_products(basis1: Basis, basis2: Basis) -> Float[Array, "m n"]:
    """Pairwise RKHS inner products between two batches of functions."""
    points = lambda b: rearrange(b.x, "n k d -> (n k) d")
    weights = lambda b: rearrange(b.a, "n k m -> (n k) m")

    # the outputs are independent, so their contributions just add up
    weighted_kernel = weights(basis1) @ weights(basis2).T

    # one kernel evaluation over all basis points, then sum within each function pair
    weighted_kernel = weighted_kernel * basis1.kernel(points(basis1), points(basis2))
    k1, k2 = basis1.a.shape[-2], basis2.a.shape[-2]
    pattern = "(m k1) (n k2) -> m n"
    return reduce(weighted_kernel, pattern, "sum", k1=k1, k2=k2)


@eqx.filter_jit
def rkhs_sq_distances(basis1: Basis, basis2: Basis) -> Float[Array, "m n"]:
    """Pairwise squared RKHS distances between two batches of functions."""
    sq_norms1 = jnp.diag(rkhs_inner_products(basis1, basis1))
    sq_norms2 = jnp.diag(rkhs_inner_products(basis2, basis2))
    d2 = sq_norms1[:, None] + sq_norms2[None, :]
    d2 = d2 - 2 * rkhs_inner_products(basis1, basis2)

    # cancellation can push coincident functions slightly below zero
    return jnp.maximum(d2, 0.0)


@eqx.filter_jit
def masked_covariance(
    profile: kernels.Profile,
    d2: Float[Array, "N N"],
    rho: Scalar,
    g: Scalar,
    mask: Float[Array, "N"],
) -> Float[Array, "N N"]:
    """Observation covariance with padded entries replaced by an inert identity block."""
    Koo = profile(d2 / rho) * mask[:, None] * mask[None, :]
    return Koo + jnp.diag(g * mask + (1 - mask))


@eqx.filter_jit
@partial(jax.value_and_grad, argnums=0)
def mle_loss(
    params: Float[Array, "2"],
    profile: kernels.Profile,
    d2: Float[Array, "N N"],
    ys: Float[Array, "N"],
    mask: Float[Array, "N"],
) -> Scalar:
    """Negative marginal likelihood of the lengthscale and nugget, at a fixed padded size."""
    rho, g = params[0], params[-1]
    Koo = masked_covariance(profile, d2, rho, g, mask)
    return -loglikelihood(Koo, ys, mask)[0]


class FunctionalPosterior(eqx.Module):
    """Everything predict needs, padded to a constant size so it compiles only once."""

    profile: kernels.Profile
    rho: Scalar
    nu: Scalar
    b: Scalar
    y_best: Scalar

    # observations, padded to a fixed number of functions of a fixed basis size
    basis: Basis
    mask: Float[Array, "N"]
    ys: Float[Array, "N"]

    # factorizations of nu * Koo, cached across posterior evaluations
    Koo_lu: tuple
    Koo_inv_sum: Scalar

    @eqx.filter_jit
    def predict(self, fs: list[rkhs.Function]) -> Gaussian:
        basis = Basis.stack(fs)
        d2 = lambda b1, b2: rkhs_sq_distances(b1, b2) / self.rho
        Kxx = self.nu * self.profile(d2(basis, basis))
        Kox = self.nu * self.profile(d2(self.basis, basis))

        # padded observations are not inert in the kernel, so mask them out
        Kox = Kox * self.mask[:, None]
        return gp_posterior(
            Kxx,
            Kox,
            None,
            self.ys,
            self.b,
            Koo_lu=self.Koo_lu,
            Koo_inv_sum=self.Koo_inv_sum,
        )


class FunctionalGaussianProcess(Module):
    # kernel definition
    profile: kernels.Profile = kernels.SquaredExponential()

    # model parameters
    rho: Scalar = eqx.field(default=None)
    g: Scalar = eqx.field(default=None)
    nu: Scalar = eqx.field(default=None)
    b: Scalar = eqx.field(default=None)

    # observed data
    observed_fs: list[rkhs.Function] = eqx.field(default=None)
    observed_ys: Float[Array, "n"] = eqx.field(default=None)

    # cached covariance matrix of the observed ys, and the fixed-shape posterior state
    Koo: Float[Array, "n n"] = eqx.field(default=None)
    posterior: FunctionalPosterior = eqx.field(default=None)

    @eqx.filter_jit
    def metric(self, f1: rkhs.Function, f2: rkhs.Function) -> Scalar:
        d = rkhs_sq_distances(Basis.stack([f1]), Basis.stack([f2])) ** 0.5
        return d.squeeze()

    @eqx.filter_jit
    def kernel(
        self,
        rho: Scalar,
        fs1: list[rkhs.Function] | Basis,
        fs2: list[rkhs.Function] | Basis,
    ) -> Float[Array, "m n"]:
        basis1 = fs1 if isinstance(fs1, Basis) else Basis.stack(fs1)
        basis2 = fs2 if isinstance(fs2, Basis) else Basis.stack(fs2)
        return self.profile(rkhs_sq_distances(basis1, basis2) / rho)

    def predict(self, fs: list[rkhs.Function]) -> Gaussian:
        return self.posterior.predict(fs)

    def fit(
        self,
        fs: list[rkhs.Function],
        ys: Float[Array, "n"],
        *,
        warmstart: bool = False,
        lengthscale_range: tuple[float, float] | None = None,
        nugget_range: tuple[float, float] = DEFAULT_NUGGET_RANGE,
        max_iterations: int = 100,
        ftol: float = EPS,
        gtol: float = 0.0,
        padded_size: int | None = None,
        padded_basis_size: int | None = None,
    ) -> Self:
        # pad to a stable size, so the mle loss retraces only when the padding grows
        n, N = len(ys), padded_to(len(ys), padded_size)
        basis = Basis.stack(fs, n=N, k=padded_basis_size)
        mask = jnp.zeros(N).at[:n].set(1.0)
        padded_ys = jnp.zeros(N).at[:n].set(ys)

        # precalc the metric to speedup mle calls
        d2 = rkhs_sq_distances(basis, basis)

        def verbose_loss(params: Float[Array, "2"]):
            val, grad = mle_loss(params, self.profile, d2, padded_ys, mask)
            if jnp.isnan(val) or jnp.isnan(grad).any():
                warnings.warn(f"NaN detected in loss or gradient: {params}")
            return val, grad

        # the padded functions sit at the origin, so only the real block carries distances
        auto_range = (
            auto_lengthscale_range(self.profile, d2[:n, :n])
            if lengthscale_range is None
            else None
        )

        # initialization
        nugget = min(0.1, nugget_range[1])
        if auto_range is not None:
            lengthscale_range = auto_range
            lengthscale = float(jnp.sqrt(auto_range[0] * auto_range[1]))
        else:
            lengthscale_range = lengthscale_range or FALLBACK_LENGTHSCALE_RANGE
            lengthscale = 0.9 * lengthscale_range[0] + 0.1 * lengthscale_range[1]
        if warmstart:
            nugget = self.g if self.g is not None else nugget
            lengthscale = self.rho if self.rho is not None else lengthscale
        init_params = jnp.array([lengthscale, nugget])

        # run optimization
        result = sp.optimize.minimize(
            fun=verbose_loss,
            x0=init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=[lengthscale_range, nugget_range],
            options=dict(maxiter=max_iterations, ftol=ftol, gtol=gtol),
        )

        # extract the optimal parameters and infer the rest, still at the padded size
        rho = jnp.array(result.x[0])
        g = jnp.array(result.x[-1])
        padded_Koo = masked_covariance(self.profile, d2, rho, g, mask)
        llk, b, nu = loglikelihood(padded_Koo, padded_ys, mask)

        # an identity block keeps the padding out of the real block's factorization
        scaled_Koo = nu * padded_Koo * mask[:, None] * mask[None, :]
        scaled_Koo = scaled_Koo + jnp.diag(1 - mask)
        posterior = FunctionalPosterior(
            profile=self.profile,
            rho=rho,
            nu=nu,
            b=b,
            y_best=ys.min(),
            basis=basis,
            mask=mask,
            ys=padded_ys,
            Koo_lu=jsp.linalg.lu_factor(scaled_Koo),
            Koo_inv_sum=(
                jnp.linalg.inv(scaled_Koo) * mask[:, None] * mask[None, :]
            ).sum(),
        )

        # return a new instance with the fitted parameters and observed data
        return self._replace(
            rho=rho,
            g=g,
            nu=nu,
            b=b,
            Koo=padded_Koo[:n, :n],
            posterior=posterior,
            observed_fs=fs,
            observed_ys=ys,
        )
