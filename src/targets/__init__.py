from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import vlse


class TestFunction(Protocol):
    d: int  # dimension of the input space
    m: int = 1  # number of outputs

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


from .ridge import Ridge
from .gymnasium import Pendulum
from .pinwheel import PinWheel
from .neuralnetworks import MNIST
from .brachistochrone import Brachistochrone
from .hopper import HoppingRobot

TARGET_FNS = {
    "gramacylee": lambda: Ridge(vlse.GramacyLee(normalized=True)),
    "ackley": lambda: Ridge(vlse.Ackley(d=2, normalized=True)),
    "hartmann": lambda: Ridge(vlse.Hartmann3(normalized=True)),
    "rosenbrock": lambda: Ridge(vlse.Rosenbrock(d=4, normalized=True)),
    "michalewicz": lambda: Ridge(vlse.Michalewicz(d=5, normalized=True)),
    "pendulum": Pendulum,
    "pinwheel": PinWheel,
    "brachistochrone": Brachistochrone,
    "mnist": MNIST,
    "hopper": HoppingRobot,
}


def make_target(name: str) -> TestFunction:
    """Instantiate a benchmark target by its sweep name."""
    return TARGET_FNS[name]()
