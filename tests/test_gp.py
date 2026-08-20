import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from bofus import gp, kernels, rkhs

D, M, K = 1, 3, 1


def make_functions(n: int, m: int = M, d: int = D, k: int = K, seed: int = 0):
    rng = np.random.default_rng(seed)
    rho = jnp.full((k, d), 0.2)
    fs = [
        rkhs.Function.from_xy(
            rho,
            jnp.asarray(rng.random((k, m, d))),
            jnp.asarray(rng.uniform(-1, 1, (k, m))),
        )
        for _ in range(n)
    ]
    ys = jnp.asarray(rng.standard_normal(n))
    return fs, ys


def test_extend_sq_distances_matches_full_recompute():
    """The cached-block extension used by warmstart fits must agree with a cold distance matrix."""
    surrogate = gp.FunctionalGaussianProcess(profile=kernels.squared_exponential)
    fs, _ = make_functions(12)
    basis = gp.Basis.stack(fs)
    full = gp.sq_distances(None, basis, basis)

    cached = gp.sq_distances(None, gp.Basis.stack(fs[:-2]), gp.Basis.stack(fs[:-2]))
    extended = surrogate.extend_sq_distances(basis, fs, cached)
    np.testing.assert_allclose(extended, full, atol=1e-10)


def test_gp_posterior_cached_factors_match_fresh():
    """Passing a precomputed LU factor / inverse-sum must give the same posterior as recomputing them."""
    rng = np.random.default_rng(1)
    n = 8
    xs = jnp.asarray(rng.random((n, 3)))
    ys = jnp.asarray(rng.standard_normal(n))
    profile, rho = kernels.squared_exponential, jnp.full(3, 0.3)
    Koo = profile(kernels.sq_euclidean(rho, xs)) + 1e-6 * jnp.eye(n)
    _, b, nu = gp.loglikelihood(Koo, ys)

    candidate = jnp.asarray(rng.random((1, 3)))
    Kxx = nu * profile(kernels.sq_euclidean(rho, candidate))
    Kox = nu * profile(kernels.sq_euclidean(rho, xs, candidate))
    scaled_Koo = nu * Koo

    fresh = gp.gp_posterior(Kxx, Kox, scaled_Koo, ys, b)
    Koo_lu = jsp.linalg.lu_factor(scaled_Koo)
    Koo_inv_sum = jnp.linalg.inv(scaled_Koo).sum()
    cached = gp.gp_posterior(
        Kxx, Kox, None, ys, b, Koo_lu=Koo_lu, Koo_inv_sum=Koo_inv_sum
    )

    np.testing.assert_allclose(fresh.mean, cached.mean, atol=1e-8)
    np.testing.assert_allclose(fresh.cov, cached.cov, atol=1e-8)


def test_fit_stays_finite_on_duplicated_observations():
    """A design with repeated functions makes Koo rank-deficient; fit/predict must not turn to NaN."""
    rng = np.random.default_rng(2)
    n, distinct = 24, 3
    unique_fs, _ = make_functions(distinct, seed=2)
    fs = [unique_fs[i] for i in rng.integers(0, distinct, n)]
    ys = jnp.asarray(rng.standard_normal(n))

    fitted = gp.FunctionalGaussianProcess(profile=kernels.squared_exponential).fit(
        fs, ys
    )
    marginal = fitted.predict(fs)

    assert bool(jnp.isfinite(marginal.mean).all())
    assert bool(jnp.isfinite(marginal.cov).all())


def test_fit_warmstart_extends_cold_fit_distances():
    """Fitting with a cached distance block from a subset of the data must match a cold fit."""
    fs, ys = make_functions(10, seed=3)
    surrogate = gp.FunctionalGaussianProcess(profile=kernels.squared_exponential)

    cold = surrogate.fit(fs, ys)
    previous = surrogate.fit(fs[:-1], ys[:-1])
    warm = previous.fit(fs, ys, cached_dists=previous.d2)

    np.testing.assert_allclose(cold.rho, warm.rho, rtol=1e-4)
    np.testing.assert_allclose(cold.nu, warm.nu, rtol=1e-4)
    np.testing.assert_allclose(cold.b, warm.b, rtol=1e-4)
