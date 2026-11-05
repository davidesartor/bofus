import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vlse import Ackley, GramacyLee, Hartmann3, Michalewicz, Rosenbrock

from bofus import rkhs
from targets import Box2D, ClassicControl, Mujoco, Projection, Ridge

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


CONTROL = [
    (ClassicControl, "pendulum", "closed", 3, 1),
    (ClassicControl, "pendulum", "open", 1, 1),
    (ClassicControl, "mountain_car", "closed", 2, 1),
    (Mujoco, "reacher", "closed", 10, 2),
    (Mujoco, "hopper", "open", 1, 3),
    (Mujoco, "hopper", "closed", 11, 3),
    (Mujoco, "pusher", "closed", 23, 7),
    (Mujoco, "ant", "open", 1, 8),
    (Mujoco, "humanoid", "open", 1, 17),
    (Mujoco, "humanoid_standup", "open", 1, 17),
    (Mujoco, "ant", "closed", 105, 8),
    (Mujoco, "humanoid", "closed", 348, 17),
    (Mujoco, "inverted_pendulum", "open", 1, 1),
    (Box2D, "lunar_lander", "open", 1, 2),
    (Box2D, "lunar_lander", "closed", 8, 2),
    (Box2D, "bipedal_walker", "open", 1, 4),
    (Box2D, "bipedal_walker", "closed", 24, 4),
    (Box2D, "bipedal_walker_hardcore", "closed", 24, 4),
]


@pytest.mark.parametrize("problem, name, mode, d, k", CONTROL)
def test_control_tasks_are_finite_and_deterministic(problem, name, mode, d, k):
    t = problem(name, mode, n_rollouts=2, max_episode_steps=20)
    assert (t.d, t.k, t.optimum) == (d, k, None)
    f = jax.tree.map(lambda z: z[0], random_functions(1, d, k=k))
    y = t(f)
    assert bool(jnp.isfinite(y))
    assert float(y) == float(t(f))


@pytest.mark.parametrize("problem, name, mode, d, k", CONTROL[::2])
def test_control_tasks_reject_wrong_number_of_outputs(problem, name, mode, d, k):
    t = problem(name, mode, n_rollouts=1, max_episode_steps=5)
    f = jax.tree.map(lambda z: z[0], random_functions(1, d, k=k + 1))
    with pytest.raises(ValueError, match=f"{k} outputs"):
        t(f)


def test_bounded_observations_map_linearly_to_unit_cube():
    t = ClassicControl("mountain_car", "closed")
    np.testing.assert_allclose(t.normalize_observation(t.obs_low), 0.0)
    np.testing.assert_allclose(t.normalize_observation(t.obs_high), 1.0)


def test_unknown_task_and_mode_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        Mujoco("car_racing")
    with pytest.raises(ValueError, match="unknown"):
        Box2D("car_racing")
    with pytest.raises(ValueError, match="mode"):
        ClassicControl("pendulum", "feedback")
