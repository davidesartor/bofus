"""Gymnasium classic control tasks with continuous actions."""

from targets.control import GymControl, Mode

# name -> env id, each registered both open and closed loop
ENVS = {
    "pendulum": "Pendulum-v1",
    "mountain_car": "MountainCarContinuous-v0",
}


class ClassicControl(GymControl):
    """Pendulum swing-up or continuous mountain car, open or closed loop.

    Both have bounded observation spaces, mapped linearly to [0, 1]; the pendulum
    observes (cos theta, sin theta, theta dot), so its closed loop policy has d = 3.
    """

    def __init__(self, name: str, mode: Mode = "closed", **kwargs):
        if name not in ENVS:
            raise ValueError(f"unknown classic control task {name!r}, pick from {list(ENVS)}")
        super().__init__(ENVS[name], mode, **kwargs)
