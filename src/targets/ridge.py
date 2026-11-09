from typing import Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from vlse.functions.base import TestFunction as Profile


class Ridge:
    """Profile composed with d = profile.d random linear functionals g_i = b_i + int w_i f.

    The weights w_i are trig polynomials of degree <= max_frequency, normalized in closed
    form to ||w_i||_L2 = 1. Calls on a squared exponential Euclidean RKHS function take
    exact_value, which evaluates the integrals in closed form but has to read f's basis
    points and coefficients. Anything else falls back to quadrature on self.x: with
    n_points=None a midpoint tensor grid (deterministic, O(h^2), since w_i * f is not
    band-limited), with a finite n_points the Monte Carlo estimate mean_j w_i(x_j) f(x_j),
    which converges to the same functional at the usual 1/sqrt(n) rate.
    """

    def __init__(
        self,
        profile: Profile,
        n_points: int | None = None,
        max_frequency: int = 4,
        seed: int = 0,
    ):
        self.d = d = profile.d
        self.k = 1
        self.profile = profile
        self.n_points = n_points
        k1, k2, k3, k4 = jr.split(jr.key(seed), 4)

        # d random band-limited weights w_i(x) = sum_m c_m cos(2 pi m.x + phi_m)
        modes = self.half_space_modes(d, max_frequency)
        c = jr.normal(k1, (d, len(modes))) / (1 + jnp.sum(jnp.abs(modes), axis=-1))
        phi = jr.uniform(k2, (d, len(modes)), minval=0.0, maxval=2 * jnp.pi)
        # normalize analytically to ||w_i||_L2 = 1, so both branches share one functional
        is_zero_mode = jnp.all(modes == 0, axis=-1)
        power = jnp.where(is_zero_mode, jnp.square(jnp.cos(phi)), 0.5) * jnp.square(c)
        self.c = c / jnp.sqrt(jnp.sum(power, axis=-1, keepdims=True))
        self.modes, self.phi = modes, phi

        if n_points is None:
            # tensor grid, at least Nyquist for the weights, total kept around 4096
            n = max(round(4096 ** (1 / d)), 2 * max_frequency + 1)
            axis = (jnp.arange(n) + 0.5) / n
            grid = jnp.meshgrid(*[axis] * d, indexing="ij")
            self.x = jnp.stack(grid, axis=-1).reshape(-1, d)
        else:
            self.x = jr.uniform(k4, (n_points, d), minval=0.0, maxval=1.0)
        self.w = self.weights_at(self.x)

        # sample d biases b
        self.b = jr.uniform(k3, (d,), minval=-1.0, maxval=1.0)

    @staticmethod
    def half_space_modes(d: int, max_frequency: int) -> Float[Array, "m d"]:
        """Frequencies with |m|_1 <= max_frequency, keeping only one of each +-m pair."""
        axes = [jnp.arange(-max_frequency, max_frequency + 1)] * d
        modes = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
        modes = modes[jnp.sum(jnp.abs(modes), axis=-1) <= max_frequency]
        leading = modes[jnp.arange(len(modes)), jnp.argmax(modes != 0, axis=-1)]
        return modes[leading >= 0]

    def weights_at(self, x: Float[Array, "n d"]) -> Float[Array, "d n"]:
        return jnp.einsum(
            "im,imn->in",
            self.c,
            jnp.cos(2 * jnp.pi * self.modes @ x.T + self.phi[..., None]),
        )

    @eqx.filter_jit
    def __call__(self, f: Callable[[Float[Array, "d"]], Float[Array, "k"]]) -> Scalar:
        """Profile of the quadrature estimates of g_i = b_i + int w_i f, squashed to [0, 1]."""
        g = self.b + jnp.mean(self.w * jax.vmap(f)(self.x).squeeze(-1), axis=-1)
        return self.profile(jax.nn.sigmoid(g))
