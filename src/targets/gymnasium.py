"""Gymnasium continuous-control tasks as functional objectives, closed or open loop."""

from typing import Callable
from jaxtyping import Array, Float, Scalar

import numpy as np
import jax.numpy as jnp
import gymnasium as gym

# episode lengths in steps, long enough for the task to be solvable (seconds at each env's dt)
HORIZONS = {
    "Pendulum-v1": 200,  # 10 s: swing up, then hold
    "MountainCarContinuous-v0": 250,  # 8 s: a good policy reaches the flag in 2-4 s
    "LunarLanderContinuous-v3": 400,  # 8 s: a controlled landing
    "BipedalWalker-v3": 500,  # 10 s: about ten strides
    "BipedalWalkerHardcore-v3": 500,  # 10 s
    "InvertedPendulum-v5": 200,  # 8 s: uncontrolled, it falls within a second
    "InvertedDoublePendulum-v5": 200,  # 10 s
    "Reacher-v5": 50,  # 1 s
    "Swimmer-v5": 300,  # 12 s: several strokes
    "Hopper-v5": 500,  # 4 s: about ten hops
    "Walker2d-v5": 500,  # 4 s: several strides
    "HalfCheetah-v5": 200,  # 10 s: gait period 0.3-0.5 s
    "Pusher-v5": 100,  # 5 s
    "Ant-v5": 200,  # 10 s
    "Humanoid-v5": 300,  # 4.5 s: a few strides before a fall would show
    "HumanoidStandup-v5": 600,  # 9 s: getting up takes several seconds
}

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


def gym_make(env_id: str, **kwargs) -> gym.Env:
    """gym.make with the task's horizon and observation settings."""
    return gym.make(
        env_id,
        max_episode_steps=HORIZONS[env_id],
        **OBSERVATION_KWARGS.get(env_id, {}),
        **kwargs,
    )


def observation_statistics(env_id: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and floored std of observations under uniform random actions."""
    rng = np.random.default_rng(0)
    observations = []
    for i in range(n):
        env = gym_make(env_id)
        obs, _ = env.reset(seed=10_000 + i)
        observations.append(obs)
        done = False
        while not done:
            action = rng.uniform(env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, _ = env.step(action)
            observations.append(obs)
            done = terminated or truncated
    observations = np.stack(observations)
    return observations.mean(0), np.maximum(observations.std(0), 1e-2)


class VideoRecorder(gym.Wrapper):
    """Pipe every other frame, at a third of the resolution, straight into one H.264 mp4 per episode (VS Code's preview plays no webm)."""

    def __init__(
        self, env: gym.Env, video_dir: str, frame_stride: int = 2, pixel_stride: int = 3
    ):
        super().__init__(env)
        self.video_dir = video_dir
        self.frame_stride, self.pixel_stride = frame_stride, pixel_stride
        self.writer, self.episode, self.step_count = None, 0, 0

    def reset(self, **kwargs):
        self.close_writer()
        out = self.env.reset(**kwargs)
        self.step_count = 0
        self.capture()
        return out

    def step(self, action):
        out = self.env.step(action)
        self.step_count += 1
        if self.step_count % self.frame_stride == 0:
            self.capture()
        return out

    def capture(self):
        frame = np.ascontiguousarray(
            self.env.render()[:: self.pixel_stride, :: self.pixel_stride]
        )
        if self.writer is None:
            import imageio_ffmpeg

            height, width, _ = frame.shape
            self.writer = imageio_ffmpeg.write_frames(
                f"{self.video_dir}/episode_{self.episode}.mp4",
                (width, height),
                fps=self.env.metadata.get("render_fps", 30) / self.frame_stride,
                codec="libx264",
                quality=None,
                pix_fmt_out="yuv420p",
                macro_block_size=2,
                ffmpeg_log_level="error",
                output_params=[
                    "-crf",
                    "30",
                    "-preset",
                    "veryfast",
                    "-movflags",
                    "+faststart",
                    "-an",
                ],
            )
            self.writer.send(None)
        self.writer.send(frame)

    def close_writer(self):
        if self.writer is not None:
            self.writer.close()
            self.writer, self.episode = None, self.episode + 1

    def close(self):
        self.close_writer()
        super().close()


def make_env(env_id: str, video_dir: str | None) -> gym.Env:
    """Env, wrapped to record every episode to video_dir when given."""
    if video_dir is None:
        return gym_make(env_id)
    env = gym_make(env_id, render_mode="rgb_array")

    # gymnasium 1.4 swaps the model's tracking camera for a static free one unless named
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    if renderer is not None:
        import mujoco

        track = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if track != -1:
            renderer.camera_id = track
    return VideoRecorder(env, video_dir)


def normalized_time(step: int, horizon: int) -> Float[Array, "1"]:
    """Step index mapped linearly onto [-1, 1]."""
    return jnp.array([2 * step / (horizon - 1) - 1])


def rescale(v: np.ndarray, low: np.ndarray, high: np.ndarray) -> Float[Array, "d"]:
    """Box [low, high] mapped linearly onto [-1, 1], clipped."""
    return jnp.clip(2 * (v - low) / (high - low) - 1, -1, 1)


def standardize(v: np.ndarray, mean: np.ndarray, std: np.ndarray) -> Float[Array, "d"]:
    """Standardized, then tanh-squashed into (-1, 1)."""
    return jnp.tanh((v - mean) / (2 * std))


def angle(sin: float, cos: float) -> Scalar:
    """Angle in units of pi, in [-1, 1]."""
    return jnp.arctan2(sin, cos) / jnp.pi


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
        n_calibration_rollouts: int = 10,
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
        self.periodic = (
            jnp.zeros(1, dtype=bool) if open_loop else jnp.array([True, False])
        )
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    cos_theta, sin_theta, _ = obs
                    _, _, theta_dot = standardize(obs, self.obs_mean, self.obs_std)
                    theta = angle(sin_theta, cos_theta)
                    x = jnp.array([theta, theta_dot])
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


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
        self.periodic = jnp.zeros(self.d, dtype=bool)

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
                        obs, env.observation_space.low, env.observation_space.high
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


# Box2D


class LunarLanderContinuous:
    """Land a lander on the pad with main and side engines.

    Policy: (x, y, vx, vy, angle, angular velocity, 2 leg contacts) -> (main, side)
        throttle.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class BipedalWalker:
    """Walk a 2D biped across flat terrain.

    Policy: hull (angle, angular velocity, vx, vy), 2 legs (hip, knee angles and speeds,
        contact), 10 lidar -> hip and knee torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class BipedalWalkerHardcore:
    """Walk a 2D biped across stumps, pits and stairs.

    Policy: hull (angle, angular velocity, vx, vy), 2 legs (hip, knee angles and speeds,
        contact), 10 lidar -> hip and knee torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


# MuJoCo


class InvertedPendulum:
    """Balance a pole on a cart.

    Policy: (cart position, pole angle, cart velocity, pole angular velocity) -> cart
        force.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class InvertedDoublePendulum:
    """Balance a two-link pole on a cart.

    Policy: (x, 2 pole angles, v, 2 pole angular velocities, constraint force) -> cart
        force.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "InvertedDoublePendulum-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 7
        self.k = 1
        self.optimum = None
        self.periodic = (
            jnp.zeros(1, dtype=bool)
            if open_loop
            else jnp.array([False, True, True, False, False, False, False])
        )
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    _, sin_1, sin_2, cos_1, cos_2, *_ = obs
                    x, _, _, _, _, v, omega_1, omega_2, force = standardize(
                        obs, self.obs_mean, self.obs_std
                    )
                    theta_1 = angle(sin_1, cos_1)
                    theta_2 = angle(sin_2, cos_2)
                    x = jnp.array([x, theta_1, theta_2, v, omega_1, omega_2, force])
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Reacher:
    """Move a two-joint arm's tip to a random target.

    Policy: (2 joint angles, target xy, 2 joint velocities, tip-to-target xy) -> 2 joint
        torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = (
            jnp.zeros(1, dtype=bool)
            if open_loop
            else jnp.array([True, True] + [False] * 6)
        )
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    cos_1, cos_2, sin_1, sin_2, *_ = obs
                    _, _, _, _, target_x, target_y, omega_1, omega_2, dx, dy = (
                        standardize(obs, self.obs_mean, self.obs_std)
                    )
                    theta_1 = angle(sin_1, cos_1)
                    theta_2 = angle(sin_2, cos_2)
                    x = jnp.array(
                        [theta_1, theta_2, target_x, target_y, omega_1, omega_2, dx, dy]
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Swimmer:
    """Propel a three-link swimmer forward through viscous fluid.

    Policy: (tip angle, 2 joint angles, vx, vy, tip and 2 joint angular velocities) -> 2
        joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Hopper:
    """Hop a one-legged robot forward.

    Policy: (height, torso angle, 3 joint angles, 6 velocities) -> 3 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Walker2d:
    """Walk a 2D two-legged robot forward.

    Policy: (height, torso angle, 6 joint angles, 9 velocities) -> 6 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class HalfCheetah:
    """Run a 2D cheetah forward.

    Policy: (height, torso angle, 6 joint angles, 9 velocities) -> 6 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Pusher:
    """Push an object to a goal with a seven-joint arm.

    Policy: (7 joint angles, 7 joint velocities, tip, object and goal xyz) -> 7 joint
        torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
        open_loop: bool = False,
        video_dir: str | None = None,
    ):
        self.env_id = "Pusher-v5"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 23
        self.k = 7
        self.optimum = None
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Ant:
    """Walk a four-legged ant forward.

    Policy: (height, torso quaternion, 8 joint angles, 14 velocities) -> 8 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Humanoid:
    """Walk a 3D humanoid forward without falling.

    Policy: (22 positions, 23 velocities) -> 17 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class HumanoidStandup:
    """Raise a 3D humanoid from lying down to standing.

    Policy: (22 positions, 23 velocities) -> 17 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 5,
        n_calibration_rollouts: int = 10,
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
        self.periodic = jnp.zeros(self.d, dtype=bool)
        self.obs_mean, self.obs_std = observation_statistics(
            self.env_id, n_calibration_rollouts
        )

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
                    x = standardize(obs, self.obs_mean, self.obs_std)
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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


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
