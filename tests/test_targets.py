import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from vlse import Ackley, GramacyLee, Hartmann3, Michalewicz, Rosenbrock

from test_acquisition import sample_functions
import gymnasium as gym

from targets import Projection, Ridge
from targets.gymnasium import ENVS

PROFILES = [
    GramacyLee(normalized=True),
    Ackley(d=2, normalized=True),
    Hartmann3(normalized=True),
    Rosenbrock(d=4, normalized=True),
    Michalewicz(d=5, normalized=True),
]


def random_functions(n, d, k=1, m=4, seed=0):
    l_range = (jnp.asarray(0.01), jnp.asarray(1.0))
    return sample_functions(jr.key(seed), (n, k, m, d), l_range)


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
    # the target squashes f with tanh, clipped since the rescaled profile touches +-1
    rescaled = lambda x: (profile((x + 1) / 2) - t.shift) / t.scale
    f = lambda x: jnp.arctanh(rescaled(x).clip(-1 + 1e-6, 1 - 1e-6))[None]
    assert float(t(f)) == pytest.approx(t.optimum, abs=1e-5)


def test_ridge_optimum_is_none_without_normalization():
    assert Ridge(Ackley(d=2)).optimum is None


CONTROL = [
    ("Pendulum-v1", 2, 1),
    ("MountainCarContinuous-v0", 2, 1),
    ("Reacher-v5", 8, 2),
    ("Hopper-v5", 11, 3),
    ("Pusher-v5", 23, 7),
    ("Ant-v5", 105, 8),
    ("Humanoid-v5", 348, 17),
    ("LunarLanderContinuous-v3", 8, 2),
    ("BipedalWalker-v3", 24, 4),
    ("BipedalWalkerHardcore-v3", 24, 4),
]


@pytest.mark.parametrize("env_id, d, k", CONTROL)
def test_control_tasks_are_finite_and_deterministic(env_id, d, k):
    t = ENVS[env_id](n_rollouts=2)
    assert (t.d, t.k, t.optimum) == (d, k, None)
    f = jax.tree.map(lambda z: z[0], random_functions(1, d, k=k))
    y = t(f)
    assert bool(jnp.isfinite(y))
    assert float(y) == float(t(f))


@pytest.mark.parametrize(
    "env_id, k", [("Pendulum-v1", 1), ("LunarLanderContinuous-v3", 2)]
)
def test_open_loop_tasks_are_finite_and_deterministic(env_id, k):
    t = ENVS[env_id](n_rollouts=2, open_loop=True)
    assert (t.d, t.k, t.optimum) == (1, k, None)
    f = jax.tree.map(lambda z: z[0], random_functions(1, 1, k=k))
    y = t(f)
    assert bool(jnp.isfinite(y))
    assert float(y) == float(t(f))


def first_policy_input(t) -> np.ndarray:
    """Input f sees at the first step of the first rollout."""
    inputs = []

    def f(x):
        inputs.append(np.asarray(x))
        return jnp.zeros(t.k)

    t.n_rollouts = 1
    t(f)
    return inputs[0]


def test_bounded_observations_map_linearly_to_the_input_box():
    t = ENVS["MountainCarContinuous-v0"]()
    env = gym.make(t.env_id)
    obs, _ = env.reset(seed=0)
    space = env.observation_space
    expected = 2 * (obs - space.low) / (space.high - space.low) - 1
    np.testing.assert_allclose(first_policy_input(t), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "env_id, inputs, angles",
    [
        ("Pendulum-v1", ("periodic", "box"), ((1, 0),)),
        (
            "InvertedDoublePendulum-v5",
            ("box", "periodic", "periodic") + ("box",) * 4,
            ((1, 3), (2, 4)),
        ),
        ("Reacher-v5", ("periodic", "periodic") + ("box",) * 6, ((2, 0), (3, 1))),
    ],
)
def test_angle_pairs_fold_into_periodic_inputs(env_id, inputs, angles):
    t = ENVS[env_id]()
    np.testing.assert_array_equal(t.periodic, np.array(inputs) == "periodic")
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    z = first_policy_input(t)
    assert z.shape == (len(inputs),)
    assert bool(((z >= -1.0) & (z <= 1.0)).all())
    for (s, c), i in zip(angles, np.flatnonzero(t.periodic)):
        angle = np.pi * z[i]
        np.testing.assert_allclose([np.sin(angle), np.cos(angle)], [obs[s], obs[c]], atol=1e-5)
