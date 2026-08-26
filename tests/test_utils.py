import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import cho_factor, cho_solve

from bofus import utils


def random_spd(key, n):
    A = jax.random.normal(key, (n, n))
    return A @ A.T + n * jnp.eye(n)


def test_mask_covariance_keeps_unmasked_block_and_sets_identity():
    cov = random_spd(jax.random.key(0), 6)
    mask = jnp.array([True] * 4 + [False] * 2)
    masked = utils.mask_covariance(cov, mask)

    assert masked[:4, :4] == pytest.approx(cov[:4, :4])
    assert masked[4:, 4:] == pytest.approx(jnp.eye(2))
    assert float(jnp.abs(masked[:4, 4:]).max()) == 0.0


def test_masked_solve_matches_submatrix_solve():
    """Solving with the masked factorization must reproduce the unmasked subproblem."""
    cov = random_spd(jax.random.key(0), 6)
    mask = jnp.array([True] * 4 + [False] * 2)
    b = jax.random.normal(jax.random.key(1), (6,))

    chol, _ = cho_factor(utils.mask_covariance(cov, mask))
    full = cho_solve((chol, False), jnp.where(mask, b, 0.0))

    sub_chol, _ = cho_factor(cov[:4, :4])
    sub = cho_solve((sub_chol, False), b[:4])
    np.testing.assert_allclose(full[:4], sub, rtol=1e-5)
    np.testing.assert_allclose(full[4:], 0.0, atol=1e-7)


def test_lhs_stratifies_every_coordinate():
    """Each flattened coordinate must place exactly one sample in each of the n strata."""
    n, dim = 32, 4
    u = utils.latin_hypercube_sample(jax.random.key(1), (n, dim))
    assert u.shape == (n, dim)
    strata = jnp.sort(jnp.floor(u * n), axis=0)
    assert bool((strata == jnp.arange(n)[:, None]).all())


def test_lhs_stratifies_trailing_dims_jointly():
    """Stratification applies over the flattened trailing dims, whatever their shape."""
    n, shape = 16, (16, 2, 3)
    u = utils.latin_hypercube_sample(jax.random.key(2), shape)
    assert u.shape == shape
    flat = u.reshape(n, -1)
    strata = jnp.sort(jnp.floor(flat * n), axis=0)
    assert bool((strata == jnp.arange(n)[:, None]).all())


def test_rescale_maps_unit_interval_to_box():
    low, high = jnp.array([-1.0, 0.0]), jnp.array([1.0, 10.0])
    assert utils.rescale(jnp.zeros(2), low, high) == pytest.approx(low)
    assert utils.rescale(jnp.ones(2), low, high) == pytest.approx(high)
    assert utils.rescale(jnp.full(2, 0.5), low, high) == pytest.approx((low + high) / 2)
