"""Continuous-control tasks from Gymnasium (MuJoCo) as functional objectives.

The simulator is Gymnasium's own; only the policy parameterization differs:

- open loop:   f(t) -> action, with t the normalized time in [0, 1]  (d = 1)
- closed loop: f(s) -> action, with s the squashed observation in [0, 1]^d

f is expected to output normalized actions in [-1, 1] (clipped), which are
rescaled to the action space of the environment.
"""

from typing import Callable, Literal
from jaxtyping import Array, Float, Scalar

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym


class MujocoTask:
    def __init__(
        self,
        env_id: str,
        mode: Literal["open", "closed"] = "closed",
        n_rollouts: int = 5,
        max_episode_steps: int | None = None,
        n_calibration_rollouts: int = 10,
    ):
        self.env_id = env_id
        self.mode = mode
        self.n_rollouts = n_rollouts
        self.env = gym.make(env_id, max_episode_steps=max_episode_steps)
        self.horizon = self.env.spec.max_episode_steps
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        self.k = self.env.action_space.shape[0]
        self.d = 1 if mode == "open" else self.env.observation_space.shape[0]

        # observations are unbounded: standardize with statistics of random-action
        # rollouts (seeds disjoint from the evaluation ones), then squash to [0, 1]
        self.obs_mean, self.obs_std = self.observation_statistics(n_calibration_rollouts)

    def __call__(self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]]) -> Scalar:
        returns = [self.rollout(f, seed) for seed in range(self.n_rollouts)]
        return -jnp.mean(jnp.array(returns))

    def normalize_observation(self, obs: np.ndarray) -> Float[Array, "d"]:
        return jax.nn.sigmoid((obs - self.obs_mean) / self.obs_std)

    def scale_action(self, action: Float[Array, "k"]) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def rollout(self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]], seed: int) -> float:
        obs, _ = self.env.reset(seed=seed)

        # open loop: the actions do not depend on the state, so evaluate them all at once
        if self.mode == "open":
            ts = (jnp.arange(self.horizon) + 0.5) / self.horizon
            plan = np.asarray(jax.vmap(f)(ts[:, None]))

        total = 0.0
        for t in range(self.horizon):
            action = plan[t] if self.mode == "open" else f(self.normalize_observation(obs))
            obs, reward, terminated, truncated, _ = self.env.step(self.scale_action(action))
            total += float(reward)
            if terminated or truncated:
                break
        return total if np.isfinite(total) else -1e6

    def observation_statistics(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        observations = []
        for i in range(n):
            obs, _ = self.env.reset(seed=10_000 + i)
            observations.append(obs)
            for _ in range(self.horizon):
                action = rng.uniform(self.action_low, self.action_high)
                obs, _, terminated, truncated, _ = self.env.step(action)
                observations.append(obs)
                if terminated or truncated:
                    break
        observations = np.stack(observations)
        return observations.mean(0), np.maximum(observations.std(0), 1e-2)
