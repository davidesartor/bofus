"""Gymnasium MuJoCo locomotion and manipulation tasks."""

from targets.control import GymControl, Mode

# name -> env id, each registered both open and closed loop
ENVS = {
    "inverted_pendulum": "InvertedPendulum-v5",
    "inverted_double_pendulum": "InvertedDoublePendulum-v5",
    "reacher": "Reacher-v5",
    "swimmer": "Swimmer-v5",
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
    "half_cheetah": "HalfCheetah-v5",
    "pusher": "Pusher-v5",
    "ant": "Ant-v5",
    "humanoid": "Humanoid-v5",
    "humanoid_standup": "HumanoidStandup-v5",
}


class Mujoco(GymControl):
    """A Gymnasium MuJoCo task, open or closed loop.

    MuJoCo observation spaces are unbounded, so closed loop observations are
    standardized with random-action rollout statistics and squashed to [0, 1].
    """

    def __init__(self, name: str, mode: Mode = "closed", **kwargs):
        if name not in ENVS:
            raise ValueError(f"unknown MuJoCo task {name!r}, pick from {list(ENVS)}")
        super().__init__(ENVS[name], mode, **kwargs)
