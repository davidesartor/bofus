import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

from bofus.acquisition import log_expected_improvement, upper_confidence_bound


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

from bofus import gp, kernels, rkhs, utils
from bofus.acquisition import optimize_expected_improvement

L_RANGE = (jnp.array(0.05**2), jnp.array(0.4**2))
X_RANGE = (jnp.array(0.0), jnp.array(1.0))
Y_RANGE = (jnp.array(-1.0), jnp.array(1.0))


def sample_functions(key, shape):
    n, k, m, d = shape
    p = utils.latin_hypercube_sample(key, (n, k, m, 2 * d + 1))
    log_l, x, y = jnp.split(p, [d, 2 * d], axis=-1)
    log_l = utils.rescale(log_l, jnp.log(L_RANGE[0]), jnp.log(L_RANGE[1]))
    x = utils.rescale(x, *X_RANGE)
    y = utils.rescale(y, *Y_RANGE)
    return rkhs.RBFMixture.from_lxy(jnp.exp(log_l), x, y.squeeze(-1))


def make_surrogate(n=8, k=2, m=3, d=2, seed=0):
    key = jax.random.key(seed)
    fs = sample_functions(key, (n, k, m, d))
    ys = jax.random.normal(key, (n,))
    return fs, ys, gp.GaussianProcess.fit(fs, ys, profile=kernels.matern52)


def test_optimize_returns_valid_candidate():
    _, ys, surrogate = make_surrogate()
    best = optimize_expected_improvement(
        jax.random.key(3), surrogate, L_RANGE, X_RANGE, Y_RANGE
    )

    def neg_log_ei(f):
        mu, cov = surrogate.predict(jax.tree.map(lambda z: z[None], f))
        return -log_expected_improvement(
            mu.squeeze(), cov.squeeze() ** 0.5, jnp.nanmin(surrogate.y)
        )

    assert bool(jnp.isfinite(neg_log_ei(best)))
    assert bool(jnp.isfinite(best.a).all())

    # the refined candidate must stay inside the box constraints
    assert bool((best.l <= L_RANGE[1] + 1e-8).all())
    assert bool((best.x >= X_RANGE[0] - 1e-8).all())
    assert bool((best.x <= X_RANGE[1] + 1e-8).all())


def test_optimize_clips_l_floor_to_the_ambient():
    """The result must respect the floor l + l_obs - l0 > 0 even for a wide l_range."""
    k, m, d = 1, 2, 1
    key = jax.random.key(4)
    wide = (jnp.array(1e-4), jnp.array(1.0))
    fs = sample_functions(key, (8, k, m, d))
    ys = jax.random.normal(key, (8,))
    surrogate = gp.GaussianProcess.fit(fs, ys, profile=kernels.matern52)

    best = optimize_expected_improvement(key, surrogate, wide, X_RANGE, Y_RANGE)
    l_floor = float((surrogate.l0.max() - surrogate.x.l.min()) * 1.01)
    assert bool((best.l >= max(float(wide[0]), l_floor) - 1e-8).all())
    assert bool(jnp.isfinite(best.a).all())
