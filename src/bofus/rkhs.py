from typing import NamedTuple, Self
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import equinox as eqx

from . import kernels

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))

RHO_RANGE = (0.05, 0.4)  # candidate lengthscales, same range the sweeps scan


@eqx.filter_jit
def kernel(
    rho: Float[Array, "d"],
    xs1: Float[Array, "n d"],
    xs2: Float[Array, "m d"],
) -> Float[Array, "n m"]:
    """Squared exponential kernel with a per-dimension lengthscale."""
    return kernels.squared_exponential(kernels.sq_euclidean(rho, xs1, xs2))


class Function(NamedTuple):
    """f: (... d) -> (... k), each output f_i = sum_j a[i,j] k_rho[i](., x[i,j])."""

    rho: Float[Array, "k d"]  # one lengthscale per output
    x: Float[Array, "k m d"]  # basis points, one set per output
    a: Float[Array, "k m"]  # coefficients

    @property
    def d(self) -> int:
        return self.x.shape[-1]

    @property
    def k(self) -> int:
        return self.x.shape[-3]

    @property
    def m(self) -> int:
        return self.x.shape[-2]

    @eqx.filter_jit
    def __call__(self, t: Float[Array, "... d"]) -> Float[Array, "... k"]:
        ts = t.reshape(-1, self.d)  # the outputs share one flat batch of query points
        Ktx = jax.vmap(kernel, in_axes=(0, None, 0))(self.rho, ts, self.x)
        ys = jnp.einsum("knm,km->nk", Ktx, self.a)
        return ys.reshape(*t.shape[:-1], self.k)

    @classmethod
    def from_array(
        cls,
        rho: Float[Array, "k d"],
        p: Float[Array, "k m d+1"],
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (-1.0, 1.0),
        eps: float = 0.01,
    ) -> Self:
        x, y = p[..., :-1], p[..., -1]
        x = x * (x_range[1] - x_range[0]) + x_range[0]  # [0,1]->x_range
        y = y * (y_range[1] - y_range[0]) + y_range[0]  # [0,1]->y_range
        return cls.from_xy(rho, x, y, eps)

    @classmethod
    def from_xy(
        cls,
        rho: Float[Array, "k d"],
        x: Float[Array, "k m d"],
        y: Float[Array, "k m"],
        eps: float = 0.01,
    ) -> Self:
        def interpolate(rho, x, y) -> Float[Array, "m"]:
            return jnp.linalg.solve(kernel(rho, x, x) + eps * jnp.eye(len(x)), y)

        return cls(rho=rho, x=x, a=jax.vmap(interpolate)(rho, x, y))


@eqx.filter_jit
def inner_product(f1: Function, f2: Function) -> Scalar:
    """RKHS inner product, summed over the independent outputs."""
    Kxx = jax.vmap(kernel)(f1.rho, f1.x, f2.x)
    return jnp.einsum("kij,ki,kj->", Kxx, f1.a, f2.a)


@eqx.filter_jit
def ambient_inner_product(
    ambient_rho: Float[Array, "d"], f1: Function, f2: Function
) -> Scalar:
    """Inner product of squared exponential functions with their own lengthscales,
    hosted in a wider squared exponential space, see ambient.tex."""
    # diagonals of the inverse precisions A_p^-1, with A_p = diag(1 / rho_p^2)
    l0, l1, l2 = ambient_rho**2, f1.rho**2, f2.rho**2
    ls = l1 + l2 - l0  # diagonal of (A1^-1 + A2^-1 - A0^-1), per output

    # |A1|^-1/2 |A2|^-1/2 |A0|^1/2 |A1^-1 + A2^-1 - A0^-1|^-1/2
    scale = jnp.sqrt(jnp.prod(l1 * l2 / (l0 * ls), axis=-1))

    profile = lambda ls, x1, x2: kernels.squared_exponential(
        kernels.sq_euclidean(jnp.sqrt(ls), x1, x2)
    )
    Kxx = jax.vmap(profile)(ls, f1.x, f2.x)
    return jnp.einsum("k,kij,ki,kj->", scale, Kxx, f1.a, f2.a)
