import jax
import jax.numpy as jnp
import pytest

from bofus import kernels, rkhs


def random_mixture(key, k=2, m=4, d=3, l_min=0.5):
    key1, key2, key3 = jax.random.split(key, 3)
    l = jax.random.uniform(key1, (k, m, d)) + l_min
    x = jax.random.uniform(key2, (k, m, d))
    a = jax.random.normal(key3, (k, m))
    return rkhs.RBFMixture(l, x, a)


def test_mixture_interpolates_with_per_point_lengthscales():
    key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(key1, (2, 5, 3))
    y = jax.random.uniform(key2, (2, 5))
    l = jax.random.uniform(key3, (2, 5, 3)) + 0.5

    f = rkhs.RBFMixture.from_lxy(l, x, y, eps=1e-6)
    for k in range(2):
        assert f(x[k])[:, k] == pytest.approx(y[k], abs=1e-3)


def test_pad_to_leaves_evaluations_unchanged():
    """Padded atoms carry zero weight, so they must not move the function values."""
    f = random_mixture(jax.random.key(0))
    padded = f.pad_to(9)
    t = jax.random.uniform(jax.random.key(1), (7, 3))
    assert padded.a.shape[-1] == 9
    assert padded(t) == pytest.approx(f(t), abs=1e-6)


def test_ambient_inner_product_norm_is_positive():
    key1, key2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(key1, (2, 4, 3))
    l = jax.random.uniform(key2, (2, 4, 3)) + 1.0
    a = jax.random.normal(jax.random.key(2), (2, 4))
    f = rkhs.RBFMixture(l, x, a)

    l0 = jnp.full((2, 3), 0.25)  # l > l0 / 2 everywhere
    assert (rkhs.rbf_inner(l0, f, f) > 0.0).all()


def test_ambient_inner_product_matches_plain_rkhs_when_l_is_ambient():
    """With every lengthscale equal to the host's, scale=1 and ls=l0."""
    key1, key2 = jax.random.split(jax.random.key(0))
    x1 = jax.random.uniform(key1, (1, 4, 2))
    x2 = jax.random.uniform(key2, (1, 3, 2))
    a1 = jnp.ones((1, 4))
    a2 = jnp.ones((1, 3))
    l0 = jnp.full((1, 2), 0.7**2)

    f1 = rkhs.RBFMixture(jnp.broadcast_to(l0[:, None], x1.shape), x1, a1)
    f2 = rkhs.RBFMixture(jnp.broadcast_to(l0[:, None], x2.shape), x2, a2)

    Kxx = kernels.rbf(kernels.euclidean_distance(l0[0], x1[0][:, None], x2[0][None]))
    plain = jnp.einsum("ij,i,j->", Kxx, a1[0], a2[0])
    ambient = rkhs.rbf_inner(l0, f1, f2)
    assert float(ambient[0]) == pytest.approx(float(plain), rel=1e-5)


def test_ambient_inner_product_is_symmetric():
    f1 = random_mixture(jax.random.key(0))
    f2 = random_mixture(jax.random.key(1))
    l0 = jnp.full((2, 3), 0.25)
    assert rkhs.rbf_inner(l0, f1, f2) == pytest.approx(
        rkhs.rbf_inner(l0, f2, f1), rel=1e-5
    )


def test_inner_product_is_inf_outside_the_ambient_rkhs():
    """A pair with l1 + l2 - l0 <= 0 falls outside the host space."""
    f = random_mixture(jax.random.key(0), l_min=0.1)
    narrow = f._replace(l=jnp.full_like(f.l, 0.1))
    l0 = jnp.full((2, 3), 0.5)
    assert bool(jnp.isinf(rkhs.rbf_inner(l0, narrow, narrow)).all())
    assert bool(jnp.isinf(rkhs.rbf_distance(l0, narrow, narrow)).all())


def test_distance_to_itself_is_zero_with_finite_gradients():
    """Coincident functions must hit the exact-zero branch, not sqrt's cusp."""
    f = random_mixture(jax.random.key(0))
    l0 = jnp.full((2, 3), 0.25)
    assert rkhs.rbf_distance(l0, f, f) == pytest.approx(0.0, abs=1e-3)

    grads = jax.grad(lambda f: rkhs.rbf_distance(l0, f, f).sum())(f)
    assert bool(jax.tree.all(jax.tree.map(lambda g: jnp.isfinite(g).all(), grads)))


def test_distance_is_symmetric_and_invariant_to_atom_order():
    f1 = random_mixture(jax.random.key(0))
    f2 = random_mixture(jax.random.key(1))
    l0 = jnp.full((2, 3), 0.25)

    d12 = rkhs.rbf_distance(l0, f1, f2)
    d21 = rkhs.rbf_distance(l0, f2, f1)
    assert d12 == pytest.approx(d21)

    shuffled = jax.tree.map(lambda z: z[:, ::-1], f2)
    assert rkhs.rbf_distance(l0, f1, shuffled) == pytest.approx(d12, rel=1e-5)


def test_distance_matches_norm_of_pointwise_difference():
    """d(f1, f2)^2 = <f1,f1> + <f2,f2> - 2<f1,f2> by construction."""
    f1 = random_mixture(jax.random.key(0))
    f2 = random_mixture(jax.random.key(1))
    l0 = jnp.full((2, 3), 0.25)

    d2 = (
        rkhs.rbf_inner(l0, f1, f1)
        + rkhs.rbf_inner(l0, f2, f2)
        - 2.0 * rkhs.rbf_inner(l0, f1, f2)
    )
    assert rkhs.rbf_distance(l0, f1, f2) ** 2 == pytest.approx(d2, rel=1e-5)
