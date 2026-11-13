"""Gymnasium continuous-control tasks as functional objectives, closed or open loop."""

from typing import Callable
from jaxtyping import Array, Float, Scalar

import numpy as np
import jax.numpy as jnp
import gymnasium as gym

# observations reduced to joint positions and velocities, dropping contact and inertia terms
HUMANOID_POSITIONS_VELOCITIES = dict(
    include_cinert_in_observation=False,
    include_cvel_in_observation=False,
    include_qfrc_actuator_in_observation=False,
    include_cfrc_ext_in_observation=False,
)
OBSERVATION_KWARGS = {
    "Ant-v5": dict(include_cfrc_ext_in_observation=False),
    "Humanoid-v5": HUMANOID_POSITIONS_VELOCITIES,
    "HumanoidStandup-v5": HUMANOID_POSITIONS_VELOCITIES,
}


def make_env(env_id: str, video_dir: str | None) -> gym.Env:
    """Env, wrapped to record every episode to video_dir when given."""
    if video_dir is None:
        return gym.make(env_id, **OBSERVATION_KWARGS.get(env_id, {}))
    env = gym.make(
        env_id, render_mode="rgb_array", **OBSERVATION_KWARGS.get(env_id, {})
    )
    return gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda _: True)


def normalized_time(step: int, horizon: int) -> Float[Array, "1"]:
    """Step index mapped linearly onto [-1, 1]."""
    return jnp.array([2 * step / (horizon - 1) - 1])


def rescale(v: np.ndarray, low: np.ndarray, high: np.ndarray) -> Float[Array, "d"]:
    """Box [low, high] mapped linearly onto [-1, 1], clipped."""
    return jnp.clip(2 * (v - low) / (high - low) - 1, -1, 1)


def wrap(theta: float) -> float:
    """Angle wrapped into [-pi, pi]."""
    return np.arctan2(np.sin(theta), np.cos(theta))


def squash(u: Float[Array, "k"], low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Unbounded u tanh-squashed into the box [low, high]."""
    return np.asarray(low + (jnp.tanh(u) + 1) * 0.5 * (high - low), dtype=np.float64)


# Classic Control


class Pendulum:
    """Swing up and balance a pendulum.

    Policy: (angle, angular velocity) -> torque.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Pendulum-v1"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 2
        self.k = 1
        self.optimum = None
        self.periodic = jnp.array((False,) if open_loop else (True, False))
        self.low, self.high = np.array(
            [
                (-np.pi, np.pi),  # angle
                (-8, 8),  # angular velocity, clipped by the env
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    cos_theta, sin_theta, theta_dot = obs
                    x = rescale(
                        np.array([np.arctan2(sin_theta, cos_theta), theta_dot]),
                        self.low,
                        self.high,
                    )
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class MountainCarContinuous:
    """Drive an underpowered car up a hill.

    Policy: (position, velocity) -> force.
    """

    def __init__(
        self, n_rollouts: int = 5, open_loop: bool = False, video_dir: str | None = None
    ):
        self.env_id = "MountainCarContinuous-v0"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 2
        self.k = 1
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (-1.2, 0.45),  # position, the episode ends at the flag
                (-0.07, 0.07),  # velocity, clipped by the env
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(obs, self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


# Box2D


class LunarLanderContinuous:
    """Land a lander on the pad with main and side engines.

    Policy: (x, y, vx, vy, angle, angular velocity, 2 leg contacts) -> (main, side)
        throttle.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "LunarLanderContinuous-v3"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 8
        self.k = 2
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (-1, 1),  # x, out of view beyond
                (-0.6, 1.6),  # y, from below the pad to the spawn height
                (-3, 3),  # vx
                (-2.5, 2.5),  # vy
                (-np.pi, np.pi),  # angle
                (-3, 3),  # angular velocity
                (0, 1),  # left leg contact
                (0, 1),  # right leg contact
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(obs, self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class BipedalWalker:
    """Walk a 2D biped across flat terrain.

    Policy: hull (angle, angular velocity, vx, vy), 2 legs (hip, knee angles and speeds,
        contact), 10 lidar -> hip and knee torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "BipedalWalker-v3"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 24
        self.k = 4
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (-np.pi, np.pi),  # hull angle
                (-1, 1),  # hull angular velocity, normalized by the env
                (-1, 1),  # vx, normalized by the env
                (-1, 1),  # vy, normalized by the env
                (-0.8, 1.1),  # hip angle, joint limits
                (-1, 1),  # hip speed, normalized by the env
                (-0.6, 0.9),  # knee angle + 1, joint limits
                (-1, 1),  # knee speed, normalized by the env
                (0, 1),  # contact
                (-0.8, 1.1),  # hip angle, joint limits
                (-1, 1),  # hip speed, normalized by the env
                (-0.6, 0.9),  # knee angle + 1, joint limits
                (-1, 1),  # knee speed, normalized by the env
                (0, 1),  # contact
                *[(0, 1)] * 10,  # lidar fractions
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(obs, self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class BipedalWalkerHardcore:
    """Walk a 2D biped across stumps, pits and stairs.

    Policy: hull (angle, angular velocity, vx, vy), 2 legs (hip, knee angles and speeds,
        contact), 10 lidar -> hip and knee torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "BipedalWalkerHardcore-v3"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 24
        self.k = 4
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (-np.pi, np.pi),  # hull angle
                (-1, 1),  # hull angular velocity, normalized by the env
                (-1, 1),  # vx, normalized by the env
                (-1, 1),  # vy, normalized by the env
                (-0.8, 1.1),  # hip angle, joint limits
                (-1, 1),  # hip speed, normalized by the env
                (-0.6, 0.9),  # knee angle + 1, joint limits
                (-1, 1),  # knee speed, normalized by the env
                (0, 1),  # contact
                (-0.8, 1.1),  # hip angle, joint limits
                (-1, 1),  # hip speed, normalized by the env
                (-0.6, 0.9),  # knee angle + 1, joint limits
                (-1, 1),  # knee speed, normalized by the env
                (0, 1),  # contact
                *[(0, 1)] * 10,  # lidar fractions
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(obs, self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


# MuJoCo


class InvertedPendulum:
    """Balance a pole on a cart.

    Policy: (cart position, pole angle, cart velocity, pole angular velocity) -> cart
        force.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "InvertedPendulum-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 4
        self.k = 1
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (-1, 1),  # cart position, rail limits
                (-0.2, 0.2),  # pole angle, the episode ends beyond
                (-6, 6),  # cart velocity
                (-8, 8),  # pole angular velocity
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(obs, self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class InvertedDoublePendulum:
    """Balance a two-link pole on a cart.

    Policy: (x, 2 pole angles, v, 2 pole angular velocities) -> cart force.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "InvertedDoublePendulum-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 6
        self.k = 1
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (-1, 1),  # cart position, rail limits
                (-0.8411, 0.8411),  # pole 1 angle, tip below y = 1 ends the episode
                (-1.1714, 1.1714),  # pole 2 angle relative to pole 1, idem
                (-10, 10),  # cart velocity, clipped by the env
                (-10, 10),  # pole 1 angular velocity, idem
                (-10, 10),  # pole 2 angular velocity, idem
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x, sin_1, sin_2, cos_1, cos_2, v, omega_1, omega_2, _ = obs
                    theta_1 = np.arctan2(sin_1, cos_1)
                    theta_2 = np.arctan2(sin_2, cos_2)
                    x = rescale(
                        np.array([x, theta_1, theta_2, v, omega_1, omega_2]),
                        self.low,
                        self.high,
                    )
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class Reacher:
    """Move a two-joint arm's tip to a random target.

    Policy: (2 joint angles, target xy, 2 joint velocities, tip-to-target xy) -> 2 joint
        torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Reacher-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 8
        self.k = 2
        self.optimum = None
        self.periodic = jnp.array(
            (False,) if open_loop else (True, True) + (False,) * 6
        )
        self.low, self.high = np.array(
            [
                (-np.pi, np.pi),  # joint 1 angle
                (-np.pi, np.pi),  # joint 2 angle
                (-0.2, 0.2),  # target x, sampled in the disc of radius 0.2
                (-0.2, 0.2),  # target y
                (-130, 130),  # joint 1 angular velocity
                (-50, 50),  # joint 2 angular velocity
                (-0.41, 0.41),  # tip-to-target x, arm reach 0.21 + target 0.2
                (-0.41, 0.41),  # tip-to-target y
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    cos_1, cos_2, sin_1, sin_2, *rest = obs
                    theta_1 = np.arctan2(sin_1, cos_1)
                    theta_2 = np.arctan2(sin_2, cos_2)
                    x = rescale(
                        np.array([theta_1, theta_2, *rest]), self.low, self.high
                    )
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class Swimmer:
    """Propel a three-link swimmer forward through viscous fluid.

    Policy: (tip angle, 2 joint angles, vx, vy, tip and 2 joint angular velocities) -> 2
        joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Swimmer-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 8
        self.k = 2
        self.optimum = None
        self.periodic = jnp.array((False,) if open_loop else (True,) + (False,) * 7)
        self.low, self.high = np.array(
            [
                (-np.pi, np.pi),  # heading, wrapped
                (-1.75, 1.75),  # joint 1 angle, joint limits
                (-1.75, 1.75),  # joint 2 angle
                (-5, 5),  # vx
                (-5, 5),  # vy
                (-5, 5),  # heading angular velocity
                (-10, 10),  # joint 1 angular velocity
                (-10, 10),  # joint 2 angular velocity
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(np.r_[wrap(obs[0]), obs[1:]], self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class Hopper:
    """Hop a one-legged robot forward.

    Policy: (height, torso angle, 3 joint angles, 6 velocities) -> 3 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Hopper-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 11
        self.k = 3
        self.optimum = None
        self.periodic = jnp.array(
            (False,) if open_loop else (False, True) + (False,) * 9
        )
        self.low, self.high = np.array(
            [
                (0.05, 2.5),  # torso height
                (-np.pi, np.pi),  # torso angle, wrapped
                (-2.8, 0.2),  # thigh, joint limits
                (-2.8, 0.2),  # leg
                (-1, 1),  # foot
                *[(-10, 10)] * 6,  # velocities, clipped by the env
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(
                        np.r_[obs[0], wrap(obs[1]), obs[2:]], self.low, self.high
                    )
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class Walker2d:
    """Walk a 2D two-legged robot forward.

    Policy: (height, torso angle, 6 joint angles, 9 velocities) -> 6 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Walker2d-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 17
        self.k = 6
        self.optimum = None
        self.periodic = jnp.array(
            (False,) if open_loop else (False, True) + (False,) * 15
        )
        self.low, self.high = np.array(
            [
                (0.05, 2.0),  # torso height
                (-np.pi, np.pi),  # torso angle, wrapped
                (-3.2, 0.6),  # thigh, joint limits
                (-3.2, 0.6),  # leg
                (-1.6, 1.6),  # foot
                (-3.2, 0.6),  # left thigh, joint limits
                (-3.2, 0.6),  # leg
                (-1.6, 1.6),  # foot
                *[(-10, 10)] * 9,  # velocities, clipped by the env
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(
                        np.r_[obs[0], wrap(obs[1]), obs[2:]], self.low, self.high
                    )
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class HalfCheetah:
    """Run a 2D cheetah forward.

    Policy: (height, torso angle, 6 joint angles, 9 velocities) -> 6 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "HalfCheetah-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 17
        self.k = 6
        self.optimum = None
        self.periodic = jnp.array(
            (False,) if open_loop else (False, True) + (False,) * 15
        )
        self.low, self.high = np.array(
            [
                (-0.65, 0.6),  # torso height offset
                (-np.pi, np.pi),  # torso angle, wrapped
                (-0.67, 1.2),  # back thigh, joint limits
                (-0.95, 0.95),  # back shin
                (-0.55, 0.95),  # back foot
                (-1.15, 0.85),  # front thigh
                (-1.35, 1.0),  # front shin
                (-0.65, 0.65),  # front foot
                (-15, 15),  # vx
                (-6, 6),  # vz
                (-15, 15),  # torso angular velocity
                *[(-30, 30)] * 6,  # joint velocities
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(
                        np.r_[obs[0], wrap(obs[1]), obs[2:]], self.low, self.high
                    )
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class Pusher:
    """Push an object to a goal with a seven-joint arm.

    Policy: (7 joint angles, 7 joint velocities, tip xyz, object xy) -> 7 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Pusher-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 19
        self.k = 7
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (-2.2854, 1.7146),  # shoulder pan, joint limits
                (-0.5236, 1.3963),  # shoulder lift
                (-1.5, 1.7),  # upper arm roll
                (-2.3213, 0),  # elbow flex
                (-1.5, 1.5),  # forearm roll
                (-1.094, 0),  # wrist flex
                (-1.5, 1.5),  # wrist roll
                (-2, 2),  # shoulder pan velocity
                (-2, 2),  # shoulder lift velocity
                (-14, 14),  # upper arm roll velocity
                (-12, 12),  # elbow flex velocity
                (-14, 14),  # forearm roll velocity
                (-9, 9),  # wrist flex velocity
                (-14, 14),  # wrist roll velocity
                (-0.65, 0.82),  # tip x, arm reach
                (-1.42, 0.22),  # tip y
                (-0.31, 0.52),  # tip z
                (0.2, 0.7),  # object x, around its spawn and the goal
                (-0.4, 0.0),  # object y
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    # object z and goal xyz are constant
                    x = rescale(obs[:19], self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class Ant:
    """Walk a four-legged ant forward.

    Policy: (height, torso quaternion, 8 joint angles, 14 velocities) -> 8 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Ant-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 27
        self.k = 8
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (0.25, 2.0),  # torso height
                *[(-1, 1)] * 4,  # torso quaternion
                (-0.7, 0.7),  # hip 1, joint limits
                (0.35, 1.4),  # ankle 1
                (-0.7, 0.7),  # hip 2, joint limits
                (-1.4, -0.35),  # ankle 2
                (-0.7, 0.7),  # hip 3, joint limits
                (-1.4, -0.35),  # ankle 3
                (-0.7, 0.7),  # hip 4, joint limits
                (0.35, 1.4),  # ankle 4
                (-10, 10),  # vx
                (-10, 10),  # vy
                (-6, 6),  # vz
                *[(-15, 15)] * 3,  # torso angular velocity
                *[(-20, 20)] * 8,  # joint velocities
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(obs, self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class Humanoid:
    """Walk a 3D humanoid forward without falling.

    Policy: (22 positions, 23 velocities) -> 17 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Humanoid-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 45
        self.k = 17
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (0.05, 1.8),  # torso height
                *[(-1, 1)] * 4,  # torso quaternion
                (-0.785, 0.785),  # abdomen z, joint limits
                (-1.309, 0.524),  # abdomen y
                (-0.611, 0.611),  # abdomen x
                (-0.436, 0.087),  # right hip x, joint limits
                (-1.047, 0.611),  # right hip z
                (-1.92, 0.349),  # right hip y
                (-2.793, -0.035),  # right knee
                (-0.436, 0.087),  # left hip x, joint limits
                (-1.047, 0.611),  # left hip z
                (-1.92, 0.349),  # left hip y
                (-2.793, -0.035),  # left knee
                (-1.484, 1.047),  # right shoulder 1
                (-1.484, 1.047),  # right shoulder 2
                (-1.571, 0.873),  # right elbow
                (-1.047, 1.484),  # left shoulder 1
                (-1.047, 1.484),  # left shoulder 2
                (-1.571, 0.873),  # left elbow
                *[(-6, 6)] * 3,  # torso velocity
                *[(-15, 15)] * 3,  # torso angular velocity
                *[(-20, 20)] * 3,  # abdomen velocities
                (-20, 20),  # hip x velocity
                (-20, 20),  # hip z velocity
                (-30, 30),  # hip y velocity
                (-40, 40),  # knee velocity
                (-20, 20),  # hip x velocity
                (-20, 20),  # hip z velocity
                (-30, 30),  # hip y velocity
                (-40, 40),  # knee velocity
                *[(-40, 40)] * 6,  # arm velocities
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(obs, self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


class HumanoidStandup:
    """Raise a 3D humanoid from lying down to standing.

    Policy: (22 positions, 23 velocities) -> 17 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "HumanoidStandup-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 45
        self.k = 17
        self.optimum = None
        self.periodic = jnp.array((False,) * self.d)
        self.low, self.high = np.array(
            [
                (0.05, 1.6),  # torso height
                *[(-1, 1)] * 4,  # torso quaternion
                (-0.785, 0.785),  # abdomen z, joint limits
                (-1.309, 0.524),  # abdomen y
                (-0.611, 0.611),  # abdomen x
                (-0.436, 0.087),  # right hip x, joint limits
                (-1.047, 0.611),  # right hip z
                (-1.92, 0.349),  # right hip y
                (-2.793, -0.035),  # right knee
                (-0.436, 0.087),  # left hip x, joint limits
                (-1.047, 0.611),  # left hip z
                (-2.094, 0.349),  # left hip y
                (-2.793, -0.035),  # left knee
                (-1.484, 1.047),  # right shoulder 1
                (-1.484, 1.047),  # right shoulder 2
                (-1.571, 0.873),  # right elbow
                (-1.047, 1.484),  # left shoulder 1
                (-1.047, 1.484),  # left shoulder 2
                (-1.571, 0.873),  # left elbow
                *[(-5, 5)] * 3,  # torso velocity
                *[(-15, 15)] * 3,  # torso angular velocity
                *[(-20, 20)] * 3,  # abdomen velocities
                (-20, 20),  # hip x velocity
                (-20, 20),  # hip z velocity
                (-30, 30),  # hip y velocity
                (-40, 40),  # knee velocity
                (-20, 20),  # hip x velocity
                (-20, 20),  # hip z velocity
                (-30, 30),  # hip y velocity
                (-40, 40),  # knee velocity
                *[(-40, 40)] * 6,  # arm velocities
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir)

        returns = []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    x = rescale(obs, self.low, self.high)
                u = f(x, **f_kwargs)
                action = squash(u, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                done = terminated or truncated
                step += 1
            returns.append(total)
        env.close()
        return -jnp.mean(jnp.array(returns))


ENVS = {
    "Pendulum-v1": Pendulum,
    "MountainCarContinuous-v0": MountainCarContinuous,
    "LunarLanderContinuous-v3": LunarLanderContinuous,
    "BipedalWalker-v3": BipedalWalker,
    "BipedalWalkerHardcore-v3": BipedalWalkerHardcore,
    "InvertedPendulum-v5": InvertedPendulum,
    "InvertedDoublePendulum-v5": InvertedDoublePendulum,
    "Reacher-v5": Reacher,
    "Swimmer-v5": Swimmer,
    "Hopper-v5": Hopper,
    "Walker2d-v5": Walker2d,
    "HalfCheetah-v5": HalfCheetah,
    "Pusher-v5": Pusher,
    "Ant-v5": Ant,
    "Humanoid-v5": Humanoid,
    "HumanoidStandup-v5": HumanoidStandup,
}
