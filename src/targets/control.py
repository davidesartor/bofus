"""Gymnasium continuous-control environments as functional objectives.

Shared by the classic control and MuJoCo problem classes. The simulator is
Gymnasium's own; only the policy parameterization differs:

- open loop:   f(t) -> action, with t the normalized time in [0, 1]  (d = 1)
- closed loop: f(s) -> action, with s the observation mapped to [0, 1]^d, angles
  observed as (sin, cos) pairs folded into one periodic input

f outputs normalized actions in (-1, 1), rescaled to the action space.
"""

from typing import Callable, Literal, Sequence
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
    Each (sin, cos) index pair in angles is replaced by the angle as a fraction of
    a turn, declared a periodic input. A rollout that goes non-finite stops there,
    keeping the return collected so far.
    """

    optimum: float | None = None

    def __init__(
        self,
        env_id: str,
        mode: Mode = "closed",
        n_rollouts: int = 5,
        max_episode_steps: int | None = None,
        n_calibration_rollouts: int = 10,
        angles: Sequence[tuple[int, int]] = (),
    ):
        if mode not in ("open", "closed"):
            raise ValueError(f"mode must be 'open' or 'closed', got {mode!r}")
        self.env_id = env_id
        self.mode = mode
        self.n_rollouts = n_rollouts
        self.env = gym.make(env_id, max_episode_steps=max_episode_steps)
        self.horizon = self.env.spec.max_episode_steps
        self.action_low = self.env.action_space.low.astype(np.float64)
        self.action_high = self.env.action_space.high.astype(np.float64)

        self.k = self.env.action_space.shape[0]
        self.outputs = ("tanh",) * self.k

        # each angle sits at its sin index, its cos index is dropped
        n_obs = self.env.observation_space.shape[0]
        self.angles = tuple(angles)
        sin_idx, cos_idx = {s for s, _ in self.angles}, {c for _, c in self.angles}
        self.keep = np.array([i for i in range(n_obs) if i not in cos_idx], dtype=int)
        if mode == "open":
            self.d, self.inputs = 1, ("box",)
        else:
            self.d = len(self.keep)
            self.inputs = tuple("periodic" if i in sin_idx else "box" for i in self.keep)

        low = self.env.observation_space.low.astype(np.float64)
        high = self.env.observation_space.high.astype(np.float64)
        self.bounded = np.isfinite(low) & np.isfinite(high)
        self.obs_low, self.obs_high = low, high
        if mode == "closed" and not self.bounded.all():
            self.obs_mean, self.obs_std = self.observation_statistics(
                n_calibration_rollouts
            )

    def __call__(self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]]) -> Scalar:
        returns = [self.rollout(f, seed) for seed in range(self.n_rollouts)]
        return -jnp.mean(jnp.array(returns))

    def normalize_observation(self, obs: np.ndarray) -> Float[Array, "d"]:
        span = np.where(self.bounded, self.obs_high - self.obs_low, 1.0)
        z = ((obs - np.where(self.bounded, self.obs_low, 0.0)) / span).clip(0.0, 1.0)
        if not self.bounded.all():
            squashed = jax.nn.sigmoid((obs - self.obs_mean) / self.obs_std)
            z = jnp.where(self.bounded, z, squashed)
        z = jnp.asarray(z)
        for s, c in self.angles:
            z = z.at[s].set((jnp.arctan2(obs[s], obs[c]) / (2 * jnp.pi)) % 1.0)
        return z[self.keep]

    def scale_action(self, action: Float[Array, "k"]) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        span = self.action_high - self.action_low
        return self.action_low + (action + 1.0) * 0.5 * span

    def rollout(
        self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]], seed: int
    ) -> float:
        obs, _ = self.env.reset(seed=seed)

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
            obs, reward, terminated, truncated, _ = self.env.step(self.scale_action(action))
            if not (np.isfinite(reward) and np.isfinite(obs).all()):
                break
            total += float(reward)
            if terminated or truncated:
                break
        return total

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
