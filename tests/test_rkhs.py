import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import equinox as eqx

from bofus import kernels, rkhs

BOX = (-1.0, 1.0)


def random_mixture(key, k=2, m=4, d=3, l_min=0.5):
    key1, key2, key3 = jr.split(key, 3)
    l = jr.uniform(key1, (k, m, d)) + l_min
    x = jr.uniform(key2, (k, m, d), minval=-1.0, maxval=1.0)
    a = jr.normal(key3, (k, m))
    return rkhs.RBFMixture(l, x, a)


def test_mixture_sums_per_point_rbf_atoms():
    """The mixture is the amplitude-weighted RBF sum with each atom's own lengthscale."""
    f = random_mixture(jr.key(0))
    t = jr.uniform(jr.key(1), (7, 3))

    expected = jnp.einsum(
        "km,tkm->tk",
        f.a,
        kernels.rbf(kernels.euclidean_distance(f.l, t[:, None, None, :], f.x)),
    )
    assert jax.vmap(lambda z: f(z, BOX))(t) == pytest.approx(expected, abs=1e-6)


def test_pad_to_leaves_evaluations_unchanged():
    """Padded atoms carry zero weight, so they must not move the function values."""
    f = random_mixture(jr.key(0))
    padded = f.pad_to(jr.key(2), 9, (0.01, 1.0), BOX)
    t = jr.uniform(jr.key(1), (7, 3))
    assert padded.a.shape[-1] == 9
    evaluate = lambda g: jax.vmap(lambda z: g(z, BOX))(t)
    assert evaluate(padded) == pytest.approx(evaluate(f), abs=1e-6)


def test_ambient_inner_product_norm_is_positive():
    key1, key2 = jr.split(jr.key(0))
    x = jr.uniform(key1, (2, 4, 3))
    l = jr.uniform(key2, (2, 4, 3)) + 1.0
    a = jr.normal(jr.key(2), (2, 4))
    f = rkhs.RBFMixture(l, x, a)

    l0 = jnp.full(3, 0.25)  # l > l0 / 2 everywhere
    assert (rkhs.inner(f, f, l0, BOX) > 0.0).all()


def test_ambient_inner_product_matches_plain_rkhs_when_l_is_ambient():
    """With every lengthscale equal to the host's, the inner product is the kernel sum."""
    key1, key2 = jr.split(jr.key(0))
    x1 = jr.uniform(key1, (1, 4, 2))
    x2 = jr.uniform(key2, (1, 3, 2))
    a1 = jnp.ones((1, 4))
    a2 = jnp.ones((1, 3))
    l0 = jnp.full(2, 0.7**2)

    f1 = rkhs.RBFMixture(jnp.broadcast_to(l0, x1.shape), x1, a1)
    f2 = rkhs.RBFMixture(jnp.broadcast_to(l0, x2.shape), x2, a2)

    Kxx = kernels.rbf(kernels.euclidean_distance(l0, x1[0][:, None], x2[0][None]))
    plain = jnp.einsum("ij,i,j->", Kxx, a1[0], a2[0])
    ambient = rkhs.inner(f1, f2, l0, BOX)
    assert float(ambient[0]) == pytest.approx(float(plain), rel=1e-5)


def test_ambient_inner_product_is_symmetric():
    f1 = random_mixture(jr.key(0))
    f2 = random_mixture(jr.key(1))
    l0 = jnp.full(3, 0.25)
    assert rkhs.inner(f1, f2, l0, BOX) == pytest.approx(
        rkhs.inner(f2, f1, l0, BOX), rel=1e-5
    )


def test_inner_product_is_inf_outside_the_ambient_rkhs():
    """An atom with l <= l0 / 2 falls outside the host space."""
    f = random_mixture(jr.key(0), l_min=0.1)
    narrow = f._replace(l=jnp.full_like(f.l, 0.1))
    l0 = jnp.full(3, 0.5)
    assert bool(jnp.isinf(rkhs.inner(narrow, narrow, l0, BOX)).all())
    assert bool(jnp.isinf(rkhs.distance(narrow, narrow, l0, BOX)).all())


def test_distance_to_itself_is_zero_with_finite_gradients():
    """Coincident functions must hit the exact-zero branch, not sqrt's cusp."""
    f = random_mixture(jr.key(0))
    l0 = jnp.full(3, 0.25)
    assert rkhs.distance(f, f, l0, BOX) == pytest.approx(0.0, abs=1e-3)

    grads = eqx.filter_grad(lambda f: rkhs.distance(f, f, l0, BOX).sum())(f)
    assert bool(jax.tree.all(jax.tree.map(lambda g: jnp.isfinite(g).all(), grads)))


def test_distance_is_symmetric_and_invariant_to_atom_order():
    f1 = random_mixture(jr.key(0))
    f2 = random_mixture(jr.key(1))
    l0 = jnp.full(3, 0.25)

    d12 = rkhs.distance(f1, f2, l0, BOX)
    d21 = rkhs.distance(f2, f1, l0, BOX)
    assert d12 == pytest.approx(d21)

    shuffled = jax.tree.map(lambda z: z[:, ::-1], f2)
    assert rkhs.distance(f1, shuffled, l0, BOX) == pytest.approx(d12, rel=1e-5)


def test_distance_matches_norm_of_pointwise_difference():
    """d(f1, f2)^2 = <f1,f1> + <f2,f2> - 2<f1,f2> by construction."""
    f1 = random_mixture(jr.key(0))
    f2 = random_mixture(jr.key(1))
    l0 = jnp.full(3, 0.25)

    d2 = (
        rkhs.inner(f1, f1, l0, BOX)
        + rkhs.inner(f2, f2, l0, BOX)
        - 2.0 * rkhs.inner(f1, f2, l0, BOX)
    )
    assert rkhs.distance(f1, f2, l0, BOX) ** 2 == pytest.approx(d2, rel=1e-5)


MIXED = (True, False)
STRETCHED = (jnp.array([0.5, -3.0]), jnp.array([2.0, 1.0]))


def test_periodic_inputs_wrap():
    f = random_mixture(jr.key(0), k=2, m=4, d=2, l_min=0.05)
    t = jr.uniform(jr.key(1), (7, 2))
    shifted = t.at[:, 0].add(2.0)
    evaluate = lambda z: f(z, BOX, MIXED)
    assert jax.vmap(evaluate)(shifted) == pytest.approx(jax.vmap(evaluate)(t), abs=1e-5)

    # a wrapped atom is the sum of its images on the real line
    ns = jnp.arange(-30, 31)
    images = jnp.exp(-0.5 * (t[:, None, 0] + 2 * ns - f.x[0, 0, 0]) ** 2 / f.l[0, 0, 0])
    box = jnp.exp(-0.5 * (t[:, 1] - f.x[0, 0, 1]) ** 2 / f.l[0, 0, 1])
    atom = f._replace(a=jnp.zeros_like(f.a).at[0, 0].set(1.0))
    assert jax.vmap(lambda z: atom(z, BOX, MIXED))(t)[:, 0] == pytest.approx(
        images.sum(-1) * box, rel=1e-5
    )


def test_periodic_rkhs_inner_reproduces_point_evaluation():
    """An atom of the ambient lengthscale is the kernel section: <k(., c), f> = f(c)."""
    l0 = jnp.array([0.05, 0.08])
    f = random_mixture(jr.key(2), k=1, m=3, d=2, l_min=0.1)
    c = jnp.array([0.3, 0.6])
    section = rkhs.RBFMixture(l0[None, None, :], c[None, None, :], jnp.ones((1, 1)))
    assert rkhs.inner(section, f, l0, BOX, MIXED)[0] == pytest.approx(
        f(c, BOX, MIXED)[0], rel=1e-5
    )


@pytest.mark.parametrize("x_range", [BOX, STRETCHED])
def test_l2_inner_matches_quadrature_on_the_box(x_range):
    """Box dims integrate over [lo, hi], periodic dims over one period, both over sqrt(2 pi)."""
    f1 = random_mixture(jr.key(3), k=1, m=4, d=2, l_min=0.01)
    f2 = random_mixture(jr.key(4), k=1, m=3, d=2, l_min=0.01)
    midpoints = (2 * jnp.arange(400)[:, None] + 1) / 800
    lo, hi = (jnp.broadcast_to(z, (2,)) for z in x_range)
    axes = lo + (hi - lo) * midpoints
    grid = jnp.stack(jnp.meshgrid(axes[:, 0], axes[:, 1], indexing="ij"), -1).reshape(
        -1, 2
    )
    cell = jnp.prod((hi - lo) / 400)
    products = jax.vmap(lambda z: f1(z, x_range, MIXED)[0] * f2(z, x_range, MIXED)[0])(
        grid
    )
    quadrature = jnp.sum(products) * cell / (2 * jnp.pi)
    assert rkhs.inner(f1, f2, None, x_range, MIXED)[0] == pytest.approx(
        quadrature, rel=1e-3, abs=1e-5
    )


def test_periodic_inputs_wrap_with_the_range_period():
    f = random_mixture(jr.key(0), k=2, m=4, d=2, l_min=0.05)
    t = jr.uniform(jr.key(1), (7, 2))
    shifted = t.at[:, 0].add(STRETCHED[1][0] - STRETCHED[0][0])
    evaluate = lambda z: f(z, STRETCHED, MIXED)
    assert jax.vmap(evaluate)(shifted) == pytest.approx(jax.vmap(evaluate)(t), abs=1e-5)


def test_distance_without_l0_is_the_l2_distance():
    f1 = random_mixture(jr.key(3), k=1, m=4, d=2, l_min=0.01)
    f2 = random_mixture(jr.key(4), k=1, m=3, d=2, l_min=0.01)
    inner = lambda g1, g2: rkhs.inner(g1, g2, None, BOX, MIXED)
    d2 = inner(f1, f1) + inner(f2, f2) - 2 * inner(f1, f2)
    assert rkhs.distance(f1, f2, None, BOX, MIXED) ** 2 == pytest.approx(d2, rel=1e-5)


@pytest.mark.parametrize(
    "d,m,periodic,x_range",
    [
        (1, 1, None, BOX),
        (1, 16, None, BOX),
        (2, 16, None, BOX),
        (3, 64, None, BOX),
        (2, 16, (True, False), BOX),
        (2, 16, None, (jnp.array([0.0, -3.0]), jnp.array([0.5, 3.0]))),
    ],
)
def test_sample_has_unit_box_variance(d, m, periodic, x_range):
    lo, hi = x_range
    n = 4096
    a_range = (jnp.array(-3.0), jnp.array(3.0))
    key = jr.key(10 * m + d)

    # clipped to the ranges
    l_range = (jnp.array(0.01), jnp.array(1.0))
    f = rkhs.RBFMixture.sample(key, (n, m, d), l_range, a_range, x_range)
    assert f.l.shape == f.x.shape == (n, m, d) and f.a.shape == (n, m)
    assert bool((f.l >= l_range[0]).all() & (f.l <= l_range[1]).all())
    assert bool(((f.x >= lo) & (f.x <= hi)).all() & (jnp.abs(f.a) <= a_range[1]).all())

    # unclipped l and a: box mean of E f(x)^2 is 1 whatever m and d, up to Monte Carlo error
    unbounded = (jnp.array(-1e12), jnp.array(1e12))
    f = rkhs.RBFMixture.sample(
        key, (n, m, d), (jnp.array(1e-12), jnp.array(1e12)), unbounded, x_range
    )
    pts = jr.uniform(jr.key(99), (n, d), minval=lo, maxval=hi)
    vals = jax.vmap(lambda fi, xi: fi(xi, x_range, periodic))(f, pts)
    # periodic dims use the line energy, which underestimates wide atoms on the circle
    assert float(jnp.mean(vals**2)) == pytest.approx(
        1.0, rel=0.12 if periodic is None else 0.25
    )


def test_sample_single_atom_is_not_flat():
    f = rkhs.RBFMixture.sample(
        jr.key(0),
        (4000, 1, 1),
        (jnp.array(1e-12), jnp.array(1e12)),
        (jnp.array(-3.0), jnp.array(3.0)),
        BOX,
    )
    assert float(jnp.median(jnp.sqrt(f.l))) < 1.0
