import jax
import jax.numpy as jnp
import pytest

from bofus import kernels

PROFILES = [
    kernels.rbf,
    kernels.matern12,
    kernels.matern32,
    kernels.matern52,
]


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.__name__)
def test_profile_value_at_zero_is_one(profile):
    assert float(profile(jnp.array(0.0))) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "profile, slope_at_0",
    [
        (kernels.rbf, 0.0),
        (kernels.matern12, -1.0),  # the exact one-sided derivative of exp(-d)
        (kernels.matern32, 0.0),
        (kernels.matern52, 0.0),
    ],
    ids=["rbf", "matern12", "matern32", "matern52"],
)
def test_profile_grad_in_d_matches_analytic_slope(profile, slope_at_0):
    """Profiles are smooth in d, so their d-gradients need no patch at all."""
    grad = jax.grad(lambda d: profile(d).sum())(jnp.array(0.0))
    assert float(grad) == pytest.approx(slope_at_0)


@pytest.mark.parametrize(
    "p, profile",
    [(0, kernels.matern12), (1, kernels.matern32), (2, kernels.matern52)],
    ids=["matern12", "matern32", "matern52"],
)
def test_generic_matern_matches_closed_forms(p, profile):
    d = jnp.linspace(0.0, 5.0, 20)
    assert kernels.matern(p)(d) == pytest.approx(profile(d), abs=1e-6)


def test_generic_matern_stable_for_large_p():
    """Approaches the rbf profile as p grows, without overflowing."""
    d = jnp.linspace(0.0, 5.0, 20)
    k = kernels.matern(500)(d)
    assert bool(jnp.isfinite(k).all())
    assert k == pytest.approx(kernels.rbf(d), abs=1e-2)


def test_generic_matern_inf_distance_gives_zero_value_and_grad():
    for p in (0, 1, 2, 20):
        assert float(kernels.matern(p)(jnp.inf)) == 0.0
        grad = jax.grad(kernels.matern(p))(jnp.inf)
        assert float(grad) == 0.0


def test_distance_matches_naive():
    x1 = jax.random.uniform(jax.random.key(0), (5, 3))
    x2 = jax.random.uniform(jax.random.key(1), (4, 3))
    rho = jnp.array([0.5, 1.0, 2.0])

    naive = jnp.linalg.norm((x1[:, None] - x2[None]) / rho, axis=-1)
    d = kernels.euclidean_distance(rho**2, x1[:, None], x2[None])
    assert d == pytest.approx(naive, abs=1e-5)


def test_distance_at_coincident_points_is_exactly_zero():
    x = jnp.zeros(5)
    assert float(kernels.euclidean_distance(jnp.ones(5), x, x)) == 0.0


def test_distance_grad_at_coincident_points_is_finite():
    """The double-where keeps sqrt's cusp out of both value and gradient."""
    x = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.5]])
    l = jnp.ones(2)

    def pairwise(l, x1, x2):
        return kernels.euclidean_distance(l, x1[:, None], x2[None]).sum()

    grad_x = jax.grad(lambda x: pairwise(l, x, x))(x)
    grad_l = jax.grad(lambda l: pairwise(l, x, x))(l)
    assert bool(jnp.isfinite(grad_x).all())
    assert bool(jnp.isfinite(grad_l).all())


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.__name__)
def test_profile_grad_wrt_l_finite_with_duplicated_points(profile):
    """End-to-end distance -> profile chain, the shape that NaN'd on duplicated fits."""
    x = jnp.array([[0.0], [0.0], [1.0]])

    def loss(l):
        return profile(kernels.euclidean_distance(l, x[:, None], x[None])).sum()

    grad = jax.grad(loss)(jnp.array([1.0]))
    assert bool(jnp.isfinite(grad).all())


def test_kernel_grad_wrt_x_is_zero_at_coincident_points():
    """Squared exponential is smooth, so the true gradient at d=0 is exactly zero."""
    x = jnp.array([[0.5, 0.5]])
    l = jnp.ones(2)

    grad = jax.grad(
        lambda x1: kernels.rbf(kernels.euclidean_distance(l, x1[:, None], x[None])).sum()
    )(x)
    assert grad == pytest.approx(0.0)
