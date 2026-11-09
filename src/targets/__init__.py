from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar


class TestFunction(Protocol):
    d: int  # number of inputs
    k: int  # number of outputs

    def __call__(
        self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]]
    ) -> Scalar: ...


from .ridge import Ridge
from .projection import Projection
from .gymnasium import Pendulum
from .pinwheel import PinWheel
from .neuralnetworks import MNIST
from .brachistochrone import Brachistochrone
from .hopper import HoppingRobot
