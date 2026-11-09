from typing import Protocol, Callable
from jaxtyping import Float, Int, Array, Scalar

import jax.numpy as jnp
import gymnasium as gym


class TestFunction(Protocol):
    d: int

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


class Pendulum(TestFunction):
    d: int = 2

    def __init__(
        self,
        n_rollouts: int = 50,
        max_episode_length: int = 400,
        discount_factor: float = 0.99,
    ):
        self.n_rollouts = n_rollouts
        self.max_episode_length = max_episode_length
        self.discount_factor = discount_factor
        self.env = gym.make("Pendulum-v1")

    def __call__(self, f: Callable[[Float[Array, "2"]], Scalar]) -> Scalar:
        returns = [self.rollout(f, seed) for seed in range(self.n_rollouts)]
        return -jnp.mean(jnp.array(returns))

    def rollout(self, f: Callable[[Float[Array, "2"]], Scalar], seed: int) -> float:
        obs, _ = self.env.reset(seed=seed)
        rewards = []
        for _ in range(self.max_episode_length):
            # convert x,y to angle theta and normalize it
            x, y, v = obs
            theta = jnp.arctan2(y, x)
            theta = (theta + jnp.pi) / (2 * jnp.pi)  # [-pi, pi] -> [0, 1]

            # normalize the angular velocity
            lb = self.env.observation_space.low[-1]
            ub = self.env.observation_space.high[-1]
            v = (v - lb) / (ub - lb)  # [lb, ub] -> [0, 1]

            # assume the f gives a normalized action in [-1, 1]
            action = f(jnp.array([theta, v]))

            # scale the action to the action space of the environment
            action = (action + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.0) * 0.5 * (ub - lb)  # [0, 1] -> [lb, ub]

            # step the environment
            obs, reward, terminated, truncated, _ = self.env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        J = sum([r * (self.discount_factor**t) for t, r in enumerate(rewards)])
        return J
