import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

from bofus.acquisition import (
    log_expected_improvement,
    q_log_expected_improvement,
    upper_confidence_bound,
)


def naive_log_ei(mu, sigma, y_best):
    z = (y_best - mu) / sigma
    return jnp.log(sigma * (jsp.stats.norm.pdf(z) + z * jsp.stats.norm.cdf(z)))


def test_matches_naive_in_easy_regime():
    mu = jnp.linspace(-2.0, 2.0, 20)
    sigma, y_best = jnp.array(0.7), jnp.array(0.5)
    got = log_expected_improvement(mu, sigma, y_best)
    assert got == pytest.approx(naive_log_ei(mu, sigma, y_best), abs=1e-5)


def test_finite_and_monotone_for_extreme_z():
    """Far below the incumbent log EI stays finite and decreasing, no -inf or NaN."""
    z = -jnp.logspace(0.0, 12.0, 30)
    log_ei = log_expected_improvement(-z, jnp.array(1.0), jnp.array(0.0))
    assert bool(jnp.isfinite(log_ei).all())
    assert bool((jnp.diff(log_ei) < 0).all())


@pytest.mark.parametrize("mu", [0.0, 5.0, 1e5, 1e10], ids=lambda m: f"mu={m:g}")
def test_grad_finite_in_every_branch(mu):
    grad = jax.grad(
        lambda m: log_expected_improvement(m, jnp.array(1.0), jnp.array(0.0))
    )(jnp.array(mu))
    assert bool(jnp.isfinite(grad))


def test_grad_wrt_sigma_finite():
    for mu in (0.0, 5.0, 1e5):
        grad = jax.grad(
            lambda s: log_expected_improvement(jnp.array(mu), s, jnp.array(0.0))
        )(jnp.array(1.0))
        assert bool(jnp.isfinite(grad))


def test_broadcasts_over_batch_shapes():
    mu = jnp.zeros((4, 3))
    sigma = jnp.ones(3)
    out = log_expected_improvement(mu, sigma, jnp.array(0.0))
    assert out.shape == (4, 3)
    assert out == pytest.approx(float(naive_log_ei(0.0, 1.0, 0.0)))


@pytest.mark.parametrize("sigma", [1.0, 1e-3], ids=lambda s: f"sigma={s:g}")
def test_single_point_q_log_ei_matches_analytic(sigma):
    """The relu temperature sits far below the posterior scale, so small improvements stay EI."""
    mu, y_best = jnp.array([0.5 * sigma]), jnp.array(0.0)
    eps = jr.normal(jr.key(0), (50_000, 1))
    got = q_log_expected_improvement(mu, jnp.array([[sigma**2]]), y_best, eps)
    expected = log_expected_improvement(mu[0], jnp.array(sigma), y_best)
    assert got == pytest.approx(expected, abs=5e-2)


def test_ucb_matches_formula_and_broadcasts():
    mu = jnp.linspace(-1.0, 1.0, 5)
    sigma = jnp.ones((3, 5))
    out = upper_confidence_bound(mu, sigma, jnp.array(4.0))
    assert out.shape == (3, 5)
    assert out == pytest.approx(-mu + 2.0 * sigma)


def test_matches_vmap_over_scalars():
    mu = jnp.linspace(-5.0, 5.0, 11)
    sigma, y_best = jnp.array(0.3), jnp.array(1.0)
    batched = log_expected_improvement(mu, sigma, y_best)
    vmapped = jax.vmap(lambda m: log_expected_improvement(m, sigma, y_best))(mu)
    assert batched == pytest.approx(vmapped)


# ---- sampling and optimization over RBF mixtures ----

from bofus import gp, kernels
from bofus.acquisition import A_MAX, optimize_expected_improvement
from bofus.rkhs import X_MAX, RBFMixture

L_RANGE = (jnp.array(0.05**2), jnp.array(0.4**2))


def sample_functions(key, shape, l_range=L_RANGE):
    """Uniform mixtures on the box, lengthscales log-uniform."""
    *batch, m, d = shape
    key_l, key_x, key_a = jr.split(key, 3)
    log_l = jr.uniform(key_l, shape, minval=jnp.log(l_range[0]), maxval=jnp.log(l_range[1]))
    x = jr.uniform(key_x, shape, minval=-X_MAX, maxval=X_MAX)
    a = jr.uniform(key_a, (*batch, m), minval=-A_MAX, maxval=A_MAX)
    return RBFMixture(jnp.exp(log_l), x, a)


def make_surrogate(n=8, k=2, m=3, d=2, seed=0):
    key = jr.key(seed)
    fs = sample_functions(key, (n, k, m, d))
    ys = jr.normal(key, (n,))
    l0 = jnp.full(d, L_RANGE[0])
    return fs, ys, gp.GaussianProcess.fit(key, fs, ys, l0=l0)


def test_optimize_returns_valid_candidate():
    _, ys, surrogate = make_surrogate()
    best = optimize_expected_improvement(
        jr.key(3), surrogate, L_RANGE
    )

    def neg_log_ei(f):
        mu, cov = surrogate.predict_joint(f)
        return -log_expected_improvement(
            mu.squeeze(), cov.squeeze() ** 0.5, jnp.nanmin(surrogate.y)
        )

    assert bool(jnp.isfinite(neg_log_ei(best)))
    assert bool(jnp.isfinite(best.a).all())

    # the refined candidate must stay inside the box constraints
    assert bool((best.l <= L_RANGE[1] + 1e-8).all())
    assert bool(((best.x >= -1.0) & (best.x <= 1.0)).all())


def test_optimize_with_mixed_constraints():
    periodic = jnp.array([True, False])
    fs = sample_functions(jr.key(0), (8, 2, 3, 2))
    ys = jr.normal(jr.key(1), (8,))
    l0 = jnp.full(2, L_RANGE[0])
    surrogate = gp.GaussianProcess.fit(jr.key(3), fs, ys, l0=l0, periodic=periodic)
    best = optimize_expected_improvement(jr.key(2), surrogate, L_RANGE)
    assert bool(jnp.isfinite(surrogate.predict_joint(best)[0]).all())
    assert bool(((best.x[..., 0] >= -1.0) & (best.x[..., 0] <= 1.0)).all())
