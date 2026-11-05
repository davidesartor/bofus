"""Gymnasium Box2D tasks with continuous actions."""

from targets.control import GymControl, Mode

# name -> (env id, policy modes worth running); the lander starts with a random
# push and the hardcore terrain is random per episode, so both need feedback
ENVS = {
    "lunar_lander": ("LunarLanderContinuous-v3", ("closed",)),
    "bipedal_walker": ("BipedalWalker-v3", ("closed", "open")),
    "bipedal_walker_hardcore": ("BipedalWalkerHardcore-v3", ("closed",)),
}


class Box2D(GymControl):
    """A Gymnasium Box2D task, open or closed loop.

    Box2D observation spaces are bounded, so closed loop observations map
    linearly to [0, 1].
    """

    def __init__(self, name: str, mode: Mode = "closed", **kwargs):
        if name not in ENVS:
            raise ValueError(f"unknown Box2D task {name!r}, pick from {list(ENVS)}")
        super().__init__(ENVS[name][0], mode, **kwargs)
