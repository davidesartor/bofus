import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bofus import gp, rkhs

D, M, K = 1, 3, 1
L_RANGE = (jnp.array(0.2**2), jnp.array(1.0))
BOX = (-1.0, 1.0)


def make_functions(n: int, m: int = M, d: int = D, k: int = K, seed: int = 0):
    rng = np.random.default_rng(seed)
    l = jnp.full((n, k, m, d), 0.2**2)
    fs = rkhs.RBFMixture(
        l=l,
        x=jnp.asarray(rng.random((n, k, m, d))),
        a=jnp.asarray(rng.uniform(-1, 1, (n, k, m))),
    )
    ys = jnp.asarray(rng.standard_normal(n))
    return fs, ys


def fit(fs, ys, **kw):
    _, o, _, d = fs.l.shape
    l0 = jnp.full(d, L_RANGE[0])
    return gp.GaussianProcess.fit(jr.key(0), fs, ys, x_range=BOX, l0=l0, **kw)


def test_predict_is_scalar_output():
    fs, ys = make_functions(8)
    fitted = fit(fs, ys)
    mean, cov = fitted.predict_joint(fs)
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)


def test_posterior_interpolates_observations():
    """With a tiny nugget the posterior mean must pass near the observed values."""
    fs, ys = make_functions(8, seed=1)
    fitted = fit(fs, ys, nugget_range=jnp.array([1e-6, 1e-4]))
    mean, cov = fitted.predict_joint(fs)
    np.testing.assert_allclose(mean, ys, atol=1e-2)
    assert bool((jnp.diag(cov) >= -1e-8).all())


def test_l2_posterior_interpolates_observations():
    """Without an atom lengthscale range the GP compares functions in L2 on the box."""
    fs, ys = make_functions(8, seed=1)
    fitted = gp.GaussianProcess.fit(
        jr.key(0), fs, ys, x_range=BOX, nugget_range=jnp.array([1e-6, 1e-4])
    )
    mean, cov = fitted.predict_joint(fs)
    assert fitted.l0 is None
    np.testing.assert_allclose(mean, ys, atol=1e-2)
    assert bool((jnp.diag(cov) >= -1e-8).all())


def test_per_dimension_ambient_lengthscale():
    """An uneven per-dimension l0 changes the geometry and still fits."""
    fs, ys = make_functions(8, d=2, seed=1)
    constant_l0 = jnp.full(2, L_RANGE[0])
    uneven_l0 = L_RANGE[0] * jnp.array([1.0, 0.5])
    constant = gp.pairwise_squared_distance(constant_l0, BOX, fs)
    uneven = gp.pairwise_squared_distance(uneven_l0, BOX, fs)
    assert not np.allclose(uneven, constant, rtol=1e-3)

    fitted = gp.GaussianProcess.fit(
        jr.key(0),
        fs,
        ys,
        x_range=BOX,
        l0=uneven_l0,
        nugget_range=jnp.array([1e-6, 1e-4]),
    )
    mean, _ = fitted.predict_joint(fs)
    assert fitted.l0.shape == (2,)
    np.testing.assert_allclose(mean, ys, atol=1e-2)


def test_padded_observations_are_inert():
    """NaN-marked padded entries must not change the likelihood or the posterior."""
    fs, ys = make_functions(6, seed=2)
    pad = 4
    padded_fs = jax.tree.map(
        lambda z: jnp.concatenate([z, jnp.broadcast_to(z[:1], (pad, *z.shape[1:]))]), fs
    )
    padded_ys = jnp.concatenate([ys, jnp.full(pad, jnp.nan)])

    fitted = fit(fs, ys)

    def loglik(l0, f, y):
        K = fitted.profile(
            gp.utils.safe_sqrt(gp.pairwise_squared_distance(l0, BOX, f) @ fitted.rho)
        )
        return gp.loglikelihood(K + fitted.g * jnp.eye(len(y)), y)

    value_and_grad = jax.value_and_grad(loglik)
    value, grad = value_and_grad(fitted.l0, fs, ys)
    padded_value, padded_grad = value_and_grad(fitted.l0, padded_fs, padded_ys)
    np.testing.assert_allclose(value, padded_value, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad, padded_grad, rtol=1e-3, atol=1e-5)

    # same key, so both fits run from the same starts
    padded = fit(padded_fs, padded_ys)
    mean, cov = fitted.predict_joint(fs)
    padded_mean, padded_cov = padded.predict_joint(fs)
    np.testing.assert_allclose(mean, padded_mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cov, padded_cov, rtol=1e-4, atol=1e-5)


def test_fit_stays_finite_on_duplicated_observations():
    """A design with repeated functions makes the covariance rank-deficient; fit/predict must not turn to NaN."""
    rng = np.random.default_rng(2)
    n, distinct = 24, 3
    unique_fs, _ = make_functions(distinct, seed=2)
    fs = jax.tree.map(lambda z: z[jnp.asarray(rng.integers(0, distinct, n))], unique_fs)
    ys = jnp.asarray(rng.standard_normal(n))

    fitted = fit(fs, ys)
    mean, cov = fitted.predict_joint(fs)
    assert bool(jnp.isfinite(mean).all())
    assert bool(jnp.isfinite(cov).all())


def test_trend_and_scale_match_gls_formulas():
    """b and nu must equal the closed-form GLS / REML estimates on a dense covariance."""
    rng = np.random.default_rng(3)
    n = 8
    A = rng.standard_normal((n, n))
    K = jnp.asarray(A @ A.T + n * np.eye(n))
    y = jnp.asarray(rng.standard_normal(n))

    _, _, _, b, nu, dof = gp.factorize(K, y)
    assert float(dof) == n - 1

    Ki = np.linalg.inv(K)
    b_naive = np.sum(Ki @ np.asarray(y)) / np.sum(Ki)
    r = np.asarray(y) - b_naive
    nu_naive = r @ Ki @ r / (n - 1)
    np.testing.assert_allclose(float(b), b_naive, rtol=1e-4)
    np.testing.assert_allclose(float(nu), nu_naive, rtol=1e-4)


def test_loglikelihood_matches_naive_reml_formula():
    rng = np.random.default_rng(4)
    n = 8
    A = rng.standard_normal((n, n))
    K = jnp.asarray(A @ A.T + n * np.eye(n))
    y = jnp.asarray(rng.standard_normal(n))

    got = gp.loglikelihood(K, y)

    Ki = np.linalg.inv(K)
    b = np.sum(Ki @ np.asarray(y)) / np.sum(Ki)
    r = np.asarray(y) - b
    nu = r @ Ki @ r / (n - 1)
    logdet = np.linalg.slogdet(np.asarray(K))[1] + np.log(np.sum(Ki))
    naive = -0.5 * ((n - 1) * np.log(nu) + logdet)
    np.testing.assert_allclose(float(got), naive, rtol=1e-4)


def test_fitted_rho_downweights_uninformative_output():
    """y depends on output 0 only, so its fitted weight must dominate the noise output's."""
    rng = np.random.default_rng(5)
    n, k, m, d = 24, 2, 3, 1
    fs, _ = make_functions(n, m=m, d=d, k=k, seed=5)
    ys = fs(jnp.full(d, 0.5), BOX)[:, 0]

    fitted = fit(fs, ys)
    assert float(fitted.rho[0]) > 2.0 * float(fitted.rho[1])


def test_negligible_y_std_matches_noiseless_fit():
    """Measurement stds far below the nugget leave the fit as without them."""
    fs, ys = make_functions(8, seed=4)
    noiseless = fit(fs, ys)
    noisy = fit(fs, ys, y_std=jnp.full(8, 1e-6))
    np.testing.assert_allclose(noisy.nu, noiseless.nu, rtol=1e-2)
    for a, b in zip(noisy.predict_joint(fs), noiseless.predict_joint(fs)):
        np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-4)


def test_y_std_relaxes_interpolation():
    """A noisy observation pulls the posterior less than an exact one."""
    fs, ys = make_functions(8, seed=5)
    y_std = jnp.zeros(8).at[0].set(3.0)
    mean, var = fit(fs, ys, y_std=y_std).predict_marginal(fs)
    exact_mean, exact_var = fit(fs, ys).predict_marginal(fs)
    assert abs(mean[0] - ys[0]) > abs(exact_mean[0] - ys[0])
    assert var[0] > exact_var[0]


def test_padded_y_std_is_inert():
    """NaN y_std on padded entries changes nothing."""
    fs, ys = make_functions(6, seed=6)
    y_std = jnp.linspace(0.1, 0.5, 6)
    pad = 3
    padded_fs = jax.tree.map(
        lambda z: jnp.concatenate([z, jnp.broadcast_to(z[:1], (pad, *z.shape[1:]))]), fs
    )
    padded_ys = jnp.concatenate([ys, jnp.full(pad, jnp.nan)])
    padded_std = jnp.concatenate([y_std, jnp.full(pad, jnp.nan)])

    fitted = fit(fs, ys, y_std=y_std)
    padded = fit(padded_fs, padded_ys, y_std=padded_std)
    assert jnp.isfinite(padded.nu)
    for a, b in zip(fitted.predict_joint(fs), padded.predict_joint(fs)):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_dof_is_reml_and_gaussian_with_known_noise():
    fs, ys = make_functions(7, seed=7)
    assert float(fit(fs, ys).dof) == 6.0
    assert fit(fs, ys, y_std=jnp.full(7, 0.1)).dof is None
