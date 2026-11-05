"""Gymnasium continuous-control environments as functional objectives.

Shared by the classic control and MuJoCo problem classes. The simulator is
Gymnasium's own; only the policy parameterization differs:

- open loop:   f(t) -> action, with t the normalized time in [0, 1]  (d = 1)
- closed loop: f(s) -> action, with s the observation mapped to [0, 1]^d

f outputs normalized actions in [-1, 1] (clipped), rescaled to the action space.
"""

from typing import Callable, Literal
from jaxtyping import Array, Float, Scalar

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym

from targets import evaluate_on

Mode = Literal["open", "closed"]


class GymControl:
    """Minus the mean undiscounted return over seeded rollouts of a Gymnasium env.

    Observations with finite bounds in the observation space are mapped linearly
    to [0, 1]. Unbounded ones are standardized with statistics of random-action
    rollouts (seeds disjoint from the evaluation ones) and squashed by a sigmoid.
    A rollout that goes non-finite stops there, keeping the return collected so far.
    Every rollout runs in a freshly made env: a reused Box2D world drifts across
    resets, so the same seed would not reproduce the same return.
    """

    optimum: float | None = None

    def __init__(
        self,
        env_id: str,
        mode: Mode = "closed",
        n_rollouts: int = 5,
        max_episode_steps: int | None = None,
        n_calibration_rollouts: int = 10,
    ):
        if mode not in ("open", "closed"):
            raise ValueError(f"mode must be 'open' or 'closed', got {mode!r}")
        self.env_id = env_id
        self.mode = mode
        self.n_rollouts = n_rollouts
        self.env = self.make_env(max_episode_steps)
        self.horizon = self.env.spec.max_episode_steps
        self.action_low = self.env.action_space.low.astype(np.float64)
        self.action_high = self.env.action_space.high.astype(np.float64)

        self.k = self.env.action_space.shape[0]
        self.d = 1 if mode == "open" else self.env.observation_space.shape[0]

        low = self.env.observation_space.low.astype(np.float64)
        high = self.env.observation_space.high.astype(np.float64)
        self.bounded = np.isfinite(low) & np.isfinite(high)
        self.obs_low, self.obs_high = low, high
        if mode == "closed" and not self.bounded.all():
            self.obs_mean, self.obs_std = self.observation_statistics(
                n_calibration_rollouts
            )

    def make_env(self, max_episode_steps: int | None = None) -> gym.Env:
        return gym.make(self.env_id, max_episode_steps=max_episode_steps)

    def __call__(self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]]) -> Scalar:
        returns = [self.rollout(f, seed) for seed in range(self.n_rollouts)]
        return -jnp.mean(jnp.array(returns))

    def normalize_observation(self, obs: np.ndarray) -> Float[Array, "d"]:
        span = np.where(self.bounded, self.obs_high - self.obs_low, 1.0)
        linear = (obs - np.where(self.bounded, self.obs_low, 0.0)) / span
        if self.bounded.all():
            return jnp.asarray(linear.clip(0.0, 1.0))
        squashed = jax.nn.sigmoid((obs - self.obs_mean) / self.obs_std)
        return jnp.where(self.bounded, linear.clip(0.0, 1.0), squashed)

    def scale_action(self, action: Float[Array, "k"]) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        span = self.action_high - self.action_low
        return self.action_low + (action + 1.0) * 0.5 * span

    def rollout(
        self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]], seed: int
    ) -> float:
        env = self.make_env(self.horizon)
        obs, _ = env.reset(seed=seed)

        # open loop: the actions do not depend on the state, so evaluate them all at once
        if self.mode == "open":
            ts = (jnp.arange(self.horizon) + 0.5) / self.horizon
            plan = np.asarray(evaluate_on(f, ts[:, None], self.k))
        else:
            policy = lambda obs: f(self.normalize_observation(obs))
            evaluate_on(policy, obs[None], self.k)  # check the output shape once

        total = 0.0
        for t in range(self.horizon):
            action = plan[t] if self.mode == "open" else policy(obs)
            obs, reward, terminated, truncated, _ = env.step(self.scale_action(action))
            if not (np.isfinite(reward) and np.isfinite(obs).all()):
                break
            total += float(reward)
            if terminated or truncated:
                break
        env.close()
        return total

    def observation_statistics(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        observations = []
        for i in range(n):
            env = self.make_env(self.horizon)
            obs, _ = env.reset(seed=10_000 + i)
            observations.append(obs)
            for _ in range(self.horizon):
                action = rng.uniform(self.action_low, self.action_high)
                obs, _, terminated, truncated, _ = env.step(action)
                observations.append(obs)
                if terminated or truncated:
                    break
            env.close()
        observations = np.stack(observations)
        return observations.mean(0), np.maximum(observations.std(0), 1e-2)
