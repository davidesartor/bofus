import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vlse import Ackley, GramacyLee, Hartmann3, Michalewicz, Rosenbrock

from bofus import rkhs
from targets import Projection, Ridge

PROFILES = [
    GramacyLee(normalized=True),
    Ackley(d=2, normalized=True),
    Hartmann3(normalized=True),
    Rosenbrock(d=4, normalized=True),
    Michalewicz(d=5, normalized=True),
]


def random_functions(n, d, k=1, m=4, seed=0):
    return rkhs.RBFMixture.from_lhs(
        jax.random.key(seed),
        (n, k, m, d),
        (jnp.asarray(0.01), jnp.asarray(1.0)),
        (jnp.asarray(0.0), jnp.asarray(1.0)),
        (jnp.asarray(-1.0), jnp.asarray(1.0)),
    )


@pytest.mark.parametrize("target", [Ridge, Projection])
@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: type(p).__name__)
def test_values_are_finite_and_above_optimum(target, profile):
    t = target(profile)
    assert (t.d, t.k, t.optimum) == (profile.d, 1, 0.0)
    fs = random_functions(8, t.d)
    ys = jnp.array([t(jax.tree.map(lambda z: z[i], fs)) for i in range(8)])
    assert bool(jnp.isfinite(ys).all())
    assert bool((ys >= t.optimum).all())


@pytest.mark.parametrize("profile", PROFILES[:2], ids=lambda p: type(p).__name__)
def test_projection_reaches_optimum_on_the_rescaled_profile(profile):
    t = Projection(profile)
    f = lambda x: ((profile(x) - t.shift) / t.scale)[None]
    assert float(t(f)) == pytest.approx(t.optimum, abs=1e-5)


@pytest.mark.parametrize("target", [Ridge, Projection])
def test_wrong_number_of_outputs_is_rejected(target):
    t = target(Ackley(d=2, normalized=True))
    f = jax.tree.map(lambda z: z[0], random_functions(1, t.d, k=2))
    with pytest.raises(ValueError, match="1 outputs"):
        t(f)


def test_ridge_optimum_is_none_without_normalization():
    assert Ridge(Ackley(d=2)).optimum is None
