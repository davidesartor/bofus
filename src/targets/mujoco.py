"""Gymnasium MuJoCo locomotion and manipulation tasks."""

from targets.control import GymControl, Mode

# name -> (env id, policy modes worth running); stabilization and reaching need
# feedback, gaits can be played open loop
ENVS = {
    "inverted_pendulum": ("InvertedPendulum-v5", ("closed",)),
    "inverted_double_pendulum": ("InvertedDoublePendulum-v5", ("closed",)),
    "reacher": ("Reacher-v5", ("closed",)),
    "swimmer": ("Swimmer-v5", ("closed", "open")),
    "hopper": ("Hopper-v5", ("closed", "open")),
    "walker2d": ("Walker2d-v5", ("closed", "open")),
    "half_cheetah": ("HalfCheetah-v5", ("closed", "open")),
}


class Mujoco(GymControl):
    """A Gymnasium MuJoCo task, open or closed loop.

    MuJoCo observation spaces are unbounded, so closed loop observations are
    standardized with random-action rollout statistics and squashed to [0, 1].
    """

    def __init__(self, name: str, mode: Mode = "closed", **kwargs):
        if name not in ENVS:
            raise ValueError(f"unknown MuJoCo task {name!r}, pick from {list(ENVS)}")
        super().__init__(ENVS[name][0], mode, **kwargs)
