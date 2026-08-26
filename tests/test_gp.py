import jax
import jax.numpy as jnp
import numpy as np

from bofus import gp, rkhs

D, M, K = 1, 3, 1


def make_functions(n: int, m: int = M, d: int = D, k: int = K, seed: int = 0):
    rng = np.random.default_rng(seed)
    l = jnp.full((n, k, m, d), 0.2**2)
    fs = rkhs.RBFMixture.from_lxy(
        l,
        jnp.asarray(rng.random((n, k, m, d))),
        jnp.asarray(rng.uniform(-1, 1, (n, k, m))),
    )
    ys = jnp.asarray(rng.standard_normal(n))
    return fs, ys


def fit(fs, ys, **kw):
    return gp.GaussianProcess.fit(fs, ys, **kw)


def test_predict_is_scalar_output():
    fs, ys = make_functions(8)
    fitted = fit(fs, ys)
    mean, cov = fitted.predict(fs)
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)


def test_posterior_interpolates_observations():
    """With a tiny nugget the posterior mean must pass near the observed values."""
    fs, ys = make_functions(8, seed=1)
    fitted = fit(fs, ys, nugget_range=(1e-6, 1e-4))
    mean, cov = fitted.predict(fs)
    np.testing.assert_allclose(mean, ys, atol=1e-2)
    assert bool((jnp.diag(cov) >= -1e-8).all())


def test_padded_observations_are_inert():
    """NaN-marked padded entries must not change the posterior on the real data."""
    fs, ys = make_functions(6, seed=2)
    fitted = fit(fs, ys)

    pad = 4
    padded_fs = jax.tree.map(
        lambda z: jnp.concatenate([z, jnp.ones((pad, *z.shape[1:]))]), fs
    )
    padded = fit(padded_fs, jnp.concatenate([ys, jnp.full(pad, jnp.nan)]))

    mean, cov = fitted.predict(fs)
    padded_mean, padded_cov = padded.predict(fs)
    # the hyperparameter optimizer trajectory shifts slightly with the padded size
    np.testing.assert_allclose(mean, padded_mean, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(cov, padded_cov, rtol=3e-2, atol=1e-4)


def test_fit_stays_finite_on_duplicated_observations():
    """A design with repeated functions makes Koo rank-deficient; fit/predict must not turn to NaN."""
    rng = np.random.default_rng(2)
    n, distinct = 24, 3
    unique_fs, _ = make_functions(distinct, seed=2)
    fs = jax.tree.map(
        lambda z: z[jnp.asarray(rng.integers(0, distinct, n))], unique_fs
    )
    ys = jnp.asarray(rng.standard_normal(n))

    fitted = fit(fs, ys)
    mean, cov = fitted.predict(fs)
    assert bool(jnp.isfinite(mean).all())
    assert bool(jnp.isfinite(cov).all())


def test_trend_and_scale_match_gls_formulas():
    """b and nu must equal the closed-form GLS estimates on a dense covariance."""
    rng = np.random.default_rng(3)
    n = 8
    A = rng.standard_normal((n, n))
    K = jnp.asarray(A @ A.T + n * np.eye(n))
    y = jnp.asarray(rng.standard_normal(n))

    from jax.scipy.linalg import cho_factor

    b, nu = gp.trend_and_scale(cho_factor(K)[0], y)

    Ki = np.linalg.inv(K)
    b_naive = np.sum(Ki @ np.asarray(y)) / np.sum(Ki)
    r = np.asarray(y) - b_naive
    nu_naive = r @ Ki @ r / n
    np.testing.assert_allclose(float(b), b_naive, rtol=1e-4)
    np.testing.assert_allclose(float(nu), nu_naive, rtol=1e-4)


def test_loglikelihood_matches_naive_concentrated_formula():
    rng = np.random.default_rng(4)
    n = 8
    A = rng.standard_normal((n, n))
    K = jnp.asarray(A @ A.T + n * np.eye(n))
    y = jnp.asarray(rng.standard_normal(n))

    got = gp.loglikelihood(K, y)

    Ki = np.linalg.inv(K)
    b = np.sum(Ki @ np.asarray(y)) / np.sum(Ki)
    r = np.asarray(y) - b
    nu = r @ Ki @ r / n
    naive = -0.5 * (n * np.log(nu) + np.linalg.slogdet(np.asarray(K))[1])
    np.testing.assert_allclose(float(got), naive, rtol=1e-4)


def test_fitted_rho_downweights_uninformative_output():
    """y depends on output 0 only, so its fitted weight must dominate the noise output's."""
    rng = np.random.default_rng(5)
    n, k, m, d = 24, 2, 3, 1
    fs, _ = make_functions(n, m=m, d=d, k=k, seed=5)
    ys = fs(jnp.full(d, 0.5))[:, 0]

    fitted = fit(fs, ys)
    assert float(fitted.rho[0]) > 2.0 * float(fitted.rho[1])
