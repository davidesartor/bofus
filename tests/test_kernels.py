import jax
import jax.numpy as jnp
import pytest

from bofus import kernels

PROFILES = [
    kernels.squared_exponential,
    kernels.matern12,
    kernels.matern32,
    kernels.matern52,
]


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.__name__)
def test_profile_grad_at_zero_is_finite(profile):
    """The taylor patch exists precisely so autodiff never sees the sqrt cusp at d2=0."""
    grad = jax.grad(lambda d2: profile(d2).sum())(jnp.array(0.0))
    assert bool(jnp.isfinite(grad))


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.__name__)
def test_profile_grad_finite_when_zero_shares_a_batch(profile):
    """A coincident pair sitting next to distinct pairs must not poison their gradients either."""
    d2 = jnp.array([0.0, 0.25, 1.0, 4.0])
    grad = jax.grad(lambda d2: profile(d2).sum())(d2)
    assert bool(jnp.isfinite(grad).all())


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.__name__)
def test_profile_survives_negative_roundoff(profile):
    """The gram expansion can undershoot zero, so the guard must swallow small negatives."""
    d2 = jnp.array([-1e-14, -1e-16, 0.0])
    value = profile(d2)
    grad = jax.grad(lambda d2: profile(d2).sum())(d2)
    assert bool(jnp.isfinite(value).all()) and bool(jnp.isfinite(grad).all())
    assert value == pytest.approx(1.0)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.__name__)
def test_profile_value_at_zero_matches_formula_limit(profile):
    """The patch must reproduce k(0) = 1, not just avoid NaN."""
    assert float(profile(jnp.array(0.0))) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "profile, k_prime_0",
    [
        (kernels.squared_exponential, -0.5),
        (kernels.matern12, 0.0),  # true slope is -inf, so the patch is deliberately flat
        (kernels.matern32, -1.5),
        (kernels.matern52, -5 / 6),
    ],
    ids=["squared_exponential", "matern12", "matern32", "matern52"],
)
def test_profile_grad_at_zero_matches_analytic_slope(profile, k_prime_0):
    """In d2 the patched slope must equal k''(0)/2 of the d-space profile, not a stray zero."""
    grad = jax.grad(lambda d2: profile(d2).sum())(jnp.array(0.0))
    assert float(grad) == pytest.approx(k_prime_0)


@pytest.mark.parametrize(
    "profile, k_prime_0",
    [
        (kernels.squared_exponential, -0.5),
        (kernels.matern32, -1.5),
        (kernels.matern52, -5 / 6),
    ],
    ids=["squared_exponential", "matern32", "matern52"],
)
def test_profile_patch_matches_the_formula_just_outside_it(profile, k_prime_0):
    """The patch is a limit, not a separate kernel: it must meet the formula just outside zero."""
    d2 = jnp.array(1e-9)
    assert float(profile(d2)) == pytest.approx(1 + k_prime_0 * float(d2), rel=1e-6)


def test_sq_euclidean_grad_wrt_rho_finite_at_coincident_points():
    """No sqrt lives here any more, but a coincident pair must still not produce NaN."""
    x = jnp.array([[0.0], [1.0]])

    grad = jax.grad(lambda rho: kernels.sq_euclidean(rho, x).sum())(jnp.array([1.0]))
    assert bool(jnp.isfinite(grad).all())


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.__name__)
def test_kernel_grad_wrt_rho_finite_with_duplicated_points(profile):
    """End-to-end distance -> profile chain, the actual shape that NaN'd on duplicated fits."""
    x = jnp.array([[0.0], [0.0], [1.0]])

    grad = jax.grad(lambda rho: profile(kernels.sq_euclidean(rho, x)).sum())(
        jnp.array([1.0])
    )
    assert bool(jnp.isfinite(grad).all())


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.__name__)
@pytest.mark.parametrize("n, d", [(100, 2), (1000, 10), (2000, 50)])
def test_kernel_grad_finite_at_scale(profile, n, d):
    """Cancellation in the gram expansion grows with n and d; the guard must outrun it."""
    x = jax.random.uniform(jax.random.key(0), (n, d)) * 3.33
    rho = jnp.full(d, 0.3)

    K = profile(kernels.sq_euclidean(rho, x))
    grad = jax.grad(lambda rho: profile(kernels.sq_euclidean(rho, x)).sum())(rho)
    assert bool(jnp.isfinite(grad).all())
    assert jnp.diag(K) == pytest.approx(1.0)


@pytest.mark.parametrize("n, d", [(100, 2), (2000, 50)])
def test_sq_euclidean_self_block_diagonal_is_exactly_zero(n, d):
    """Cancellation leaves eps * |x/rho|^2 there, which matern12's sqrt blows up to its root."""
    x = jax.random.uniform(jax.random.key(0), (n, d)) * 3.33
    rho = jnp.full(d, 0.3)

    d2 = kernels.sq_euclidean(rho, x)
    assert bool((jnp.diag(d2) == 0.0).all())
