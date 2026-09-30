from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import scipy as sp
import numpy as np


class Brachistochrone:
    d: int = 1

    def __init__(
        self,
        initial_position: tuple[float, float] = (-3.0, 1.0),
        final_position: tuple[float, float] = (0.0, 0.0),
        gravity: float = 9.81,
        max_integration_subdivisions: int = 1000,
        offset_by_optimal_time: bool = True,
    ):
        self.initial_position = initial_position
        self.final_position = final_position
        self.gravity = gravity
        self.max_integration_subdivisions = max_integration_subdivisions
        self.offset_by_optimal_time = offset_by_optimal_time

        x0, y0 = self.initial_position
        x1, y1 = self.final_position
        assert y1 < y0, "The endpoint must be below the start point."
        assert x1 > x0, "The endpoint must be to the right of the start point."

    def __call__(self, f: Callable[[Float[Array, "1"]], Scalar]) -> Scalar:
        x0, y0 = self.initial_position
        x1, y1 = self.final_position

        @jax.jit
        def dtdx(x: Scalar):
            y, dydx = jax.value_and_grad(self.get_curve(f))(x)
            dydx = jnp.clip(dydx, -1e8, 1e8)
            v_squared = 2.0 * self.gravity * (y0 - y)
            v_squared = jnp.clip(v_squared, min=1e-8, max=1e8)
            dtdx = jnp.sqrt((1.0 + dydx**2) / v_squared)
            return dtdx
        
        t, error = sp.integrate.quad(
            dtdx, x0, x1, limit=self.max_integration_subdivisions
        )
        if self.offset_by_optimal_time:
            cycloid, optimal_time = self.find_brachistochrone()
            t = jnp.clip(t - optimal_time, min=0.0)
        return jnp.array(t)

    def get_curve(self, f: Callable[[Float[Array, "1"]], Scalar]):
        x0, y0 = self.initial_position
        x1, y1 = self.final_position
        fx0 = f(jnp.array([0.0])).squeeze()
        fx1 = f(jnp.array([1.0])).squeeze()

        def curve(x: Scalar):
            # prepare input for f
            x = jnp.array(x).reshape(1, 1)
            x = (x - x0) / (x1 - x0)  # [x0, x1] -> [0, 1]

            # evaluate y = f(x) + linear_offset(x)
            # offset ensures the trajectory passes through the start and end points
            offset = (1 - x) * (y0 - fx0) + x * (y1 - fx1)
            y = f(x) + offset

            # clip to bounds to avoid unfesible trajectories
            ub = self.upper_bound(x, normalized=True)
            y = jnp.clip(y, max=ub)
            return y.squeeze()

        return curve

    def upper_bound(self, x: Float[Array, "..."], normalized=False):
        # traight line connecting the start and end points
        x0, y0 = self.initial_position
        x1, y1 = self.final_position
        if not normalized:
            x = (x - x0) / (x1 - x0)
        return y0 + (y1 - y0) * x

    def find_brachistochrone(self, interp_points: int = 1000):
        # The cycloid can be parameterized implicitly as a function of the angle θ:
        #   Δx = r (θ - sin θ)
        #   Δy = -r (1 - cos θ)
        # the final angle θ1 is the one that satisfies:
        #   (θ - sin θ) / (1 - cos θ) - (X1 - X0) / (Y0 - Y1) = 0

        # find θ1 (final angle) using root finding with newton's method
        @jax.jit
        @jax.value_and_grad
        def f(angle):
            angle = angle % (2 * jnp.pi)
            return (angle - jnp.sin(angle)) / (1.0 - jnp.cos(angle)) - (x1 - x0) / (
                y0 - y1
            )

        x0, y0 = self.initial_position
        x1, y1 = self.final_position
        final_angle = sp.optimize.root_scalar(f, fprime=True, x0=2.0).root
        final_angle = final_angle % (2 * jnp.pi)

        # compute the radius r of the cycloid and the total travel time
        r = (y0 - y1) / (1.0 - np.cos(final_angle))
        omega = jnp.sqrt(self.gravity / r)
        travel_time = final_angle / omega

        # compute a dense grid of interpolation points
        angles = omega * jnp.linspace(0.0, travel_time, interp_points) ** 2
        x_vals = x0 + r * (angles - jnp.sin(angles))
        y_vals = y0 - r * (1.0 - jnp.cos(angles))

        # return cyclois as a callable and the optimal travel time
        cycloid = lambda x: jnp.interp(x, x_vals, y_vals)
        travel_time = jnp.sqrt(r / self.gravity) * final_angle
        return cycloid, travel_time
