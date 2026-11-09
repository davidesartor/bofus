from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from bofus import kernels, rkhs


class Profile(Protocol):
    """A scalar test function on the unit hypercube, e.g. any vlse function with normalized=True."""

    d: int

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]: ...


class TestFunction(Protocol):
    d: int  # dimension of the input space
    m: int = 1  # number of outputs

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


def faddeeva(z: Array, n_terms: int = 16) -> Array:
    """Scaled complex error function w(z) = exp(-z^2) erfc(-i z), for Im(z) >= 0.

    Weideman's rational approximation, accurate to float32 roundoff.
    """
    m = 2 * n_terms
    scale = jnp.sqrt(n_terms / jnp.sqrt(2.0))
    nodes = scale * jnp.tan(jnp.arange(-m + 1, m) * jnp.pi / (2 * m))
    weights = jnp.concatenate(
        [jnp.zeros(1), jnp.exp(-jnp.square(nodes)) * (scale**2 + jnp.square(nodes))]
    )
    coefficients = jnp.real(jnp.fft.fft(jnp.fft.fftshift(weights)))[1 : n_terms + 1]

    denominator = scale - 1j * z
    unit_disk = (scale + 1j * z) / denominator
    polynomial = jnp.polyval(jnp.flip(coefficients).astype(z.dtype), unit_disk)
    return polynomial / (m * denominator**2) + (1 / jnp.sqrt(jnp.pi)) / denominator


@eqx.filter_jit
def quadrature_value(
    profile: Profile,
    x: Float[Array, "n d"],
    w: Float[Array, "d n"],
    b: Float[Array, "d"],
    f: Callable[[Float[Array, "d"]], Scalar],
) -> Scalar:
    """Profile of the quadrature estimates of g_i = b_i + int w_i f, squashed to [0, 1]."""
    g = b + jnp.mean(w * jax.vmap(f)(x), axis=-1)
    return profile(jax.nn.sigmoid(g))


class Ridge(TestFunction):
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

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        if self.has_exact_value(f):
            return self.exact_value(f)
        return quadrature_value(self.profile, self.x, self.w, self.b, f)

    @staticmethod
    def has_exact_value(f: Callable[[Float[Array, "d"]], Scalar]) -> bool:
        """Whether f is an RKHS function of the one form the closed form covers."""
        return (
            isinstance(f, rkhs.Function)
            and isinstance(f.kernel.profile, kernels.SquaredExponential)
            and isinstance(f.kernel.metric, kernels.Euclidean)
            and f.kernel.m == 1
        )

    @eqx.filter_jit
    def exact_value(self, f: rkhs.Function) -> Scalar:
        """Reference value with the integrals in closed form. Not a black box functional.

        Only valid for a squared exponential profile on a Euclidean metric, where the
        kernel factorizes over dimensions and every int_0^1 exp(i w t) k(t, x_l) dt is a
        difference of complex error functions. Reads f's basis points and coefficients,
        so it is a baseline for the quadratures above, never a target to optimize.
        """
        assert isinstance(f.kernel.profile, kernels.SquaredExponential)
        assert isinstance(f.kernel.metric, kernels.Euclidean)
        rho, omega = f.kernel.rho, 2 * jnp.pi * self.modes

        # per dimension int_0^1 exp(i w t) exp(-(t - mu)^2 / (2 rho^2)) dt
        mu = f.x[:, None, :]  # (basis points, modes, d)
        bounds = jnp.stack(
            [(0 - mu) / (rho * jnp.sqrt(2)), (1 - mu) / (rho * jnp.sqrt(2))]
        )
        gaussian_transform = (
            rho
            * jnp.sqrt(jnp.pi / 2)
            * jnp.exp(1j * omega * mu)
            * jnp.diff(self.scaled_erf(bounds, omega * rho), axis=0).squeeze(0)
        )

        # pair the mode coefficients of w_i with those of f, then take the real part
        mode_integral = jnp.prod(gaussian_transform, axis=-1)  # (basis points, modes)
        phase = jnp.exp(1j * self.phi)  # (d, modes)
        a = f.a.reshape(len(f.x))
        g = self.b + jnp.real(self.c * phase * (a @ mode_integral)).sum(-1)
        return self.profile(jax.nn.sigmoid(g))

    @staticmethod
    def scaled_erf(u: Array, w: Array) -> Array:
        """exp(-w^2 / 2) erf(u - i w / sqrt(2)), written so nothing over- or underflows."""
        sign, v = jnp.where(u < 0, -1.0, 1.0), jnp.abs(u)
        wofz = faddeeva(sign * w / jnp.sqrt(2) + 1j * v)
        exponential = jnp.exp(-jnp.square(v) + 1j * sign * jnp.sqrt(2) * v * w)
        return sign * (jnp.exp(-jnp.square(w) / 2) - wofz * exponential)
