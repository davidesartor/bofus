from typing import Literal, Protocol
from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp

################################################################################
# region Kernel Profiles


class Profile(Protocol):
    """Kernel profile as a function of squared distance, which stays smooth at zero."""

    def __call__(self, d2: Float[Array, "..."]) -> Float[Array, "..."]: ...


class SquaredExponential(Profile):
    def __call__(self, d2: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.exp(-0.5 * d2)


class Matern(Profile):
    def __init__(self, nu: float):
        self.nu = nu

    def __call__(self, d2: Float[Array, "..."]) -> Float[Array, "..."]:
        # sqrt has an infinite derivative at zero, so evaluate it away from the cusp and
        # patch the value back in: matern kernels have no derivative at coincident points
        coincident = d2 <= 0
        d = jnp.sqrt(jnp.where(coincident, 1.0, d2))

        # TODO: add support for general nu
        if self.nu == 1 / 2:
            k = jnp.exp(-d)
        elif self.nu == 3 / 2:
            k = (1 + jnp.sqrt(3) * d) * jnp.exp(-jnp.sqrt(3) * d)
        elif self.nu == 5 / 2:
            k = (1 + jnp.sqrt(5) * d + 5 / 3 * d**2) * jnp.exp(-jnp.sqrt(5) * d)
        else:
            raise ValueError(f"Unsupported nu={self.nu}")
        return jnp.where(coincident, 1.0, k)


# endregion
################################################################################


################################################################################
# region Metrics


class Metric(Protocol):
    """Pairwise squared distances, so profiles never differentiate a square root."""

    def __call__(
        self,
        rho: Float[Array, "..."],
        x1: Float[Array, "n d"],
        x2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]: ...


class Euclidean(Metric):
    def __call__(
        self,
        rho: Float[Array, "#d"],
        x1: Float[Array, "n d"],
        x2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        v = (x1[:, None, :] - x2[None, :, :]) / rho
        return jnp.sum(v**2, axis=-1)


class Minkowski(Metric):
    def __init__(self, p: int | Literal["inf", "-inf"]):
        self.p = p

    def __call__(
        self,
        rho: Float[Array, "#d"],
        x1: Float[Array, "n d"],
        x2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        # define the squared distance function for a single pair of points
        def d2(a: Float[Array, "d"], b: Float[Array, "d"]) -> Scalar:
            v = (a - b) / rho
            # the norm is not differentiable at v=0 for general p, so keep the branch
            return jax.lax.cond(
                jnp.allclose(v, 0.0),
                lambda: 0.0,
                lambda: jax.numpy.linalg.norm(v, ord=self.p) ** 2,
            )

        # vectorize the distance function over pairs
        d2 = jax.vmap(d2, in_axes=(None, 0))  # vectorize over x2
        d2 = jax.vmap(d2, in_axes=(0, None))  # vectorize over x1
        return d2(x1, x2)


class Manhattan(Minkowski):
    def __init__(self):
        super().__init__(p=1)


class Chebyshev(Minkowski):
    def __init__(self):
        super().__init__(p="inf")


class Mahalanobis(Metric):
    def __init__(self, p: int | Literal["inf", "-inf"] = 2):
        self.p = p

    def __call__(
        self,
        rho: Float[Array, "d d"],
        x1: Float[Array, "n d"],
        x2: Float[Array, "m d"],
    ) -> Float[Array, "n m"]:
        # define the squared distance function for a single pair of points
        def d2(a: Float[Array, "d"], b: Float[Array, "d"]) -> Scalar:
            cov_sqrt, is_lower = jsp.linalg.cho_factor(rho)
            v = jsp.linalg.solve_triangular(cov_sqrt, a - b, lower=is_lower)
            if self.p == 2:
                return jnp.sum(v**2)

            # the norm is not differentiable at v=0 for general p, so keep the branch
            return jax.lax.cond(
                jnp.allclose(v, 0.0),
                lambda: 0.0,
                lambda: jax.numpy.linalg.norm(v, ord=self.p) ** 2,
            )

        # vectorize the distance function over pairs
        d2 = jax.vmap(d2, in_axes=(None, 0))  # vectorize over x2
        d2 = jax.vmap(d2, in_axes=(0, None))  # vectorize over x1
        return d2(x1, x2)


# endregion
################################################################################
