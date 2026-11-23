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

# shorter open-loop episodes for periodic tasks: the m bumps per action span the whole horizon,
# so a long one leaves too few for a startup phase plus a couple of gait cycles
OPEN_LOOP_HORIZONS = {
    "Pendulum-v1": 100,  # 5 s: the swing-up and a short hold
    "BipedalWalker-v3": 150,  # 3 s: about three strides
    "BipedalWalkerHardcore-v3": 150,  # 3 s
    "Swimmer-v5": 100,  # 4 s: one or two strokes
    "Hopper-v5": 150,  # 1.2 s: about three hops
    "Walker2d-v5": 200,  # 1.6 s: about two strides
    "HalfCheetah-v5": 40,  # 2 s: four to six gait periods
    "Ant-v5": 50,  # 2.5 s: a few gait periods
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

# full-horizon episodes: a fall ending the episode would make the return jump with the policy
TERMINATION_KWARGS = {
    env_id: dict(terminate_when_unhealthy=False)
    for env_id in ("Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5")
}


def gym_make(env_id: str, open_loop: bool = False, **kwargs) -> gym.Env:
    """gym.make with the task's horizon, observation and termination settings."""
    horizon = OPEN_LOOP_HORIZONS.get(env_id) if open_loop else None
    return gym.make(
        env_id,
        max_episode_steps=horizon or HORIZONS[env_id],
        **OBSERVATION_KWARGS.get(env_id, {}),
        **TERMINATION_KWARGS.get(env_id, {}),
        **kwargs,
    )


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


def make_env(env_id: str, video_dir: str | None, open_loop: bool = False) -> gym.Env:
    """Env, wrapped to record every episode to video_dir when given."""
    if video_dir is None:
        return gym_make(env_id, open_loop)
    env = gym_make(env_id, open_loop, render_mode="rgb_array")

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
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) if open_loop else (True, False)
        self.low, self.high = np.array(
            [
                (-np.pi, np.pi),  # angle
                (-8, 8),  # angular velocity, clipped by the env
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class MountainCarContinuous:
    """Drive an underpowered car up a hill.

    Policy: (position, velocity) -> force.
    """

    def __init__(
        self, n_rollouts: int = 1, open_loop: bool = False, video_dir: str | None = None
    ):
        self.env_id = "MountainCarContinuous-v0"
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 2
        self.k = 1
        self.optimum = None
        self.periodic = (False,) * self.d
        self.low, self.high = np.array(
            [
                (-1.2, 0.45),  # position, the episode ends at the flag
                (-0.07, 0.07),  # velocity, clipped by the env
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


# Box2D


class LunarLanderContinuous:
    """Land a lander on the pad with main and side engines.

    Policy: (x, y, vx, vy, angle, angular velocity, 2 leg contacts) -> (main, side)
        throttle.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one terrain and initial push for now
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
        self.periodic = (False,) * self.d
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class BipedalWalker:
    """Walk a 2D biped across flat terrain.

    Policy: hull (angle, angular velocity, vx, vy), 2 legs (hip, knee angles and speeds,
        contact), 10 lidar -> hip and knee torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one terrain for now
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
        self.periodic = (False,) * self.d
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class BipedalWalkerHardcore:
    """Walk a 2D biped across stumps, pits and stairs.

    Policy: hull (angle, angular velocity, vx, vy), 2 legs (hip, knee angles and speeds,
        contact), 10 lidar -> hip and knee torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one obstacle course for now
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
        self.periodic = (False,) * self.d
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


# MuJoCo


class InvertedPendulum:
    """Balance a pole on a cart.

    Policy: (cart position, pole angle, cart velocity, pole angular velocity) -> cart
        force.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) * self.d
        self.low, self.high = np.array(
            [
                (-1, 1),  # cart position, rail limits
                (-0.2, 0.2),  # pole angle, the episode ends beyond
                (-6, 6),  # cart velocity
                (-8, 8),  # pole angular velocity
            ]
        ).T

    def __call__(self, f: Callable[..., Float[Array, "k"]], **f_kwargs) -> Scalar:
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class InvertedDoublePendulum:
    """Balance a two-link pole on a cart.

    Policy: (x, 2 pole angles, v, 2 pole angular velocities) -> cart force.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) * self.d
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Reacher:
    """Move a two-joint arm's tip to a random target.

    Policy: (2 joint angles, target xy, 2 joint velocities, tip-to-target xy) -> 2 joint
        torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one target placement for now
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
        self.periodic = (False,) if open_loop else (True, True) + (False,) * 6
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
                # gear 200 makes |a| = 1 spin the arm at ~130 rad/s, a good reach needs ~0.02
                action = squash(
                    u, 0.1 * env.action_space.low, 0.1 * env.action_space.high
                )
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
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) if open_loop else (True,) + (False,) * 7
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Hopper:
    """Hop a one-legged robot forward.

    Policy: (height, torso angle, 3 joint angles, 6 velocities) -> 3 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) if open_loop else (False, True) + (False,) * 9
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Walker2d:
    """Walk a 2D two-legged robot forward.

    Policy: (height, torso angle, 6 joint angles, 9 velocities) -> 6 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) if open_loop else (False, True) + (False,) * 15
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class HalfCheetah:
    """Run a 2D cheetah forward.

    Policy: (height, torso angle, 6 joint angles, 9 velocities) -> 6 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) if open_loop else (False, True) + (False,) * 15
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Pusher:
    """Push an object to a goal with a seven-joint arm.

    Policy: (7 joint angles, 7 joint velocities, tip xyz, object xy) -> 7 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one object placement for now
        open_loop: bool = False,
        video_dir: str | None = None,
        progress_weight: float = 1.0,
    ):
        self.env_id = "Pusher-v5"
        self.progress_weight = progress_weight
        self.n_rollouts = n_rollouts
        self.open_loop = open_loop
        self.video_dir = video_dir
        self.d = 1 if open_loop else 19
        self.k = 7
        self.optimum = None
        self.periodic = (False,) * self.d
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

        returns, closest = [], []
        for seed in range(self.n_rollouts):
            obs, _ = env.reset(seed=seed)
            total, done, step = 0.0, False, 0
            min_object_to_goal = np.linalg.norm(obs[17:20] - obs[20:23])
            while not done:
                if self.open_loop:
                    x = normalized_time(step, env.spec.max_episode_steps)
                else:
                    # object z and goal xyz are constant
                    x = rescale(obs[:19], self.low, self.high)
                u = f(x, **f_kwargs)
                # full torque flails the arm, worse than idle; random policies at 0.25 beat idling
                action = squash(
                    u, 0.25 * env.action_space.low, 0.25 * env.action_space.high
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                if not (np.isfinite(reward) and np.isfinite(obs).all()):
                    break
                total += float(reward)
                min_object_to_goal = min(
                    min_object_to_goal, np.linalg.norm(obs[17:20] - obs[20:23])
                )
                done = terminated or truncated
                step += 1
            returns.append(total)
            closest.append(min_object_to_goal)
        env.close()

        # reward the puck's closest approach to the goal, which the averaged reward_dist barely sees
        cost = -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps
        return cost + self.progress_weight * jnp.mean(jnp.array(closest))


class Ant:
    """Walk a four-legged ant forward.

    Policy: (height, torso quaternion, 8 joint angles, 14 velocities) -> 8 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) * self.d
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class Humanoid:
    """Walk a 3D humanoid forward without falling.

    Policy: (22 positions, 23 velocities) -> 17 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) * self.d
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
        return -jnp.mean(jnp.array(returns)) / env.spec.max_episode_steps


class HumanoidStandup:
    """Raise a 3D humanoid from lying down to standing.

    Policy: (22 positions, 23 velocities) -> 17 joint torques.
    """

    def __init__(
        self,
        n_rollouts: int = 1,  # one start state for now
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
        self.periodic = (False,) * self.d
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
        env = make_env(self.env_id, self.video_dir, self.open_loop)

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
