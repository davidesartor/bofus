from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar


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
