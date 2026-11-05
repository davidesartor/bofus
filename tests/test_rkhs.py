import numpy as np
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


def test_l2_inner_matches_quadrature_on_unit_box():
    """Closed-form L2 inner product over [0, 1]^2 against a midpoint-rule grid."""
    f1 = random_mixture(jax.random.key(3), k=2, m=4, d=2, l_min=0.01)
    f2 = random_mixture(jax.random.key(4), k=2, m=3, d=2, l_min=0.01)
    n = 400
    t = (jnp.arange(n) + 0.5) / n
    grid = jnp.stack(jnp.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)
    quadrature = jnp.mean(f1.latent(grid) * f2.latent(grid), axis=0)
    assert rkhs.l2_inner(f1, f2) == pytest.approx(quadrature, rel=1e-3, abs=1e-5)


def test_l2_distance_is_a_metric_on_samples():
    f1 = random_mixture(jax.random.key(5), l_min=0.05)
    f2 = random_mixture(jax.random.key(6), l_min=0.05)
    assert rkhs.l2_distance(f1, f1) == pytest.approx(0.0, abs=1e-3)
    assert rkhs.l2_distance(f1, f2) == pytest.approx(rkhs.l2_distance(f2, f1), rel=1e-5)
    assert bool((rkhs.l2_distance(f1, f2) > 0.0).all())



def test_call_is_tanh_of_latent():
    f = random_mixture(jax.random.key(0))
    f = f._replace(a=5.0 * f.a)  # large amplitudes leave [-1, 1]
    t = jax.random.uniform(jax.random.key(1), (7, 3))
    assert f(t) == pytest.approx(jnp.tanh(f.latent(t)), abs=1e-6)
    assert bool(jnp.all(jnp.abs(f(t)) <= 1.0))


def test_gauss_inner_matches_quadrature_and_box_limit():
    f1 = random_mixture(jax.random.key(0), k=1, m=3, d=1, l_min=0.01)
    f2 = random_mixture(jax.random.key(1), k=1, m=3, d=1, l_min=0.01)
    t = jnp.linspace(0.0, 1.0, 20001)[:, None]
    w = jnp.exp(-0.5 * (t[:, 0] - 0.5) ** 2 / 0.25**2)
    prod = f1.latent(t)[:, 0] * f2.latent(t)[:, 0]
    exact = jnp.trapezoid(w * prod, t[:, 0])
    assert rkhs.l2_inner(f1, f2, sigma=0.25)[0] == pytest.approx(exact, rel=1e-3)
    wide = rkhs.l2_inner(f1, f2, sigma=1e4)
    assert wide == pytest.approx(rkhs.l2_inner(f1, f2), rel=1e-4)


def test_quad_distance_ignores_differences_in_saturation():
    """Atoms that only push further into saturation barely move tanh(f)."""
    l = jnp.full((1, 1, 1), 0.05)
    x = jnp.full((1, 1, 1), 0.5)
    f1 = rkhs.RBFMixture(l, x, jnp.full((1, 1), 20.0))
    f2 = f1._replace(a=jnp.full((1, 1), 40.0))
    f3 = f1._replace(a=jnp.full((1, 1), -20.0))
    near = rkhs.quad_distance(f1, f2)[0]
    far = rkhs.quad_distance(f1, f3)[0]
    assert near < 0.1 * far
    assert rkhs.quad_distance(f1, f1)[0] == 0.0


MIXED = rkhs.Space(inputs=("periodic", "box"), outputs=("tanh", "linear"))


def test_space_rejects_unknown_types_and_wrong_dimensions():
    with pytest.raises(ValueError, match="unknown"):
        rkhs.Space(inputs=("circle",))
    with pytest.raises(ValueError, match="declares"):
        MIXED.periodic(3)


def test_periodic_inputs_wrap_and_outputs_squash_per_dimension():
    f = random_mixture(jax.random.key(0), k=2, m=4, d=2, l_min=0.05)
    f = f._replace(a=5.0 * f.a, space=MIXED)
    t = jax.random.uniform(jax.random.key(1), (7, 2))
    shifted = t.at[:, 0].add(1.0)
    assert f(shifted) == pytest.approx(f(t), abs=1e-5)
    h, latent = f(t), f.latent(t)
    assert h[:, 0] == pytest.approx(jnp.tanh(latent[:, 0]), abs=1e-6)
    assert h[:, 1] == pytest.approx(latent[:, 1], abs=1e-6)

    # a wrapped atom is the sum of its images on the real line
    ns = jnp.arange(-30, 31)
    images = jnp.exp(-0.5 * (t[:, None, 0] + ns - f.x[0, 0, 0]) ** 2 / f.l[0, 0, 0])
    box = jnp.exp(-0.5 * (t[:, 1] - f.x[0, 0, 1]) ** 2 / f.l[0, 0, 1])
    atom = f._replace(a=jnp.zeros_like(f.a).at[0, 0].set(1.0))
    assert atom.latent(t)[:, 0] == pytest.approx(images.sum(-1) * box, rel=1e-5)


def test_periodic_rkhs_inner_reproduces_point_evaluation():
    """An atom of the ambient lengthscale is the kernel section: <k(., c), f> = f(c)."""
    l0 = jnp.array([[0.05, 0.08]])
    f = random_mixture(jax.random.key(2), k=1, m=3, d=2, l_min=0.1)._replace(space=MIXED)
    c = jnp.array([0.3, 0.6])
    section = rkhs.RBFMixture(l0[:, None, :], c[None, None, :], jnp.ones((1, 1)), MIXED)
    assert rkhs.rbf_inner(l0, section, f)[0] == pytest.approx(f.latent(c)[0], rel=1e-5)


@pytest.mark.parametrize("sigma", [None, 0.25])
def test_mixed_l2_inner_matches_quadrature(sigma):
    f1 = random_mixture(jax.random.key(3), k=1, m=4, d=2, l_min=0.01)._replace(space=MIXED)
    f2 = random_mixture(jax.random.key(4), k=1, m=3, d=2, l_min=0.01)._replace(space=MIXED)
    n = 400
    t = (jnp.arange(n) + 0.5) / n
    grid = jnp.stack(jnp.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)
    w = 1.0 if sigma is None else jnp.exp(-0.5 * (grid[:, 1] - 0.5) ** 2 / sigma**2)
    quadrature = jnp.mean(w * f1.latent(grid)[:, 0] * f2.latent(grid)[:, 0])
    inner = rkhs.l2_inner(f1, f2, sigma=sigma)[0]
    assert inner == pytest.approx(quadrature, rel=1e-3, abs=1e-5)


def test_centers_round_trip_through_unit_coordinates():
    l = jnp.array([[0.04, 0.25]])
    u = jnp.array([[0.0, 1.0]])
    x = rkhs.centers_from_unit(u, l, MIXED)
    np.testing.assert_allclose(x, [[0.0, 1.0 + 3 * 0.5]], atol=1e-6)
    u = jnp.array([[0.3, 0.7]])
    np.testing.assert_allclose(rkhs.centers_to_unit(rkhs.centers_from_unit(u, l, MIXED), l, MIXED), u, atol=1e-6)
