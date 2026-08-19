from typing import Callable
from functools import lru_cache
from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import numpy as np
import vlse.optim

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


@lru_cache(maxsize=None)
def autodiff_view(loss_fn: Callable, static_args) -> Callable:
    """Scalar view of a (value, gradient) loss, whose autodiff gradient is the returned one.

    The natural gradient losses return preconditioned gradients, so the optimizer must
    take the gradient the loss hands back instead of differentiating the value.
    """

    @jax.custom_jvp
    def scalar_loss(p, dynamic_args) -> Scalar:
        return loss_fn(p, *eqx.combine(dynamic_args, static_args))[0]

    @scalar_loss.defjvp
    def scalar_loss_jvp(primals, tangents):
        p, dynamic_args = primals
        val, grad = loss_fn(p, *eqx.combine(dynamic_args, static_args))
        return val, jnp.vdot(grad, tangents[0])

    return scalar_loss


@eqx.filter_jit
def refine_candidates(
    scalar_loss: Callable,
    candidates: Float[Array, "n p"],
    dynamic_args: tuple,
    in_axes: tuple,
    maxiter: int,
    ftol: float,
    gtol: float,
) -> tuple[Float[Array, "n p"], Float[Array, "n"]]:
    """Refine every candidate with box constrained L-BFGS-B, all in one dispatch."""
    solve = lambda c, args: vlse.optim.minimise(
        scalar_loss,
        c,
        bounds=(jnp.zeros_like(c), jnp.ones_like(c)),
        args=(args,),
        tol=gtol,
        ftol=ftol,
        max_iterations=maxiter,
    )
    states = jax.vmap(solve, in_axes=(0, in_axes))(candidates, dynamic_args)
    return states.x, states.f


def optimize_lhs_candidates(
    acquisition_loss: Callable,
    candidates: Float[Array, "n p"],
    extra_args: list | None = None,
    loss_args: tuple = (),
    max_restarts: int = 0,
    screening_loss: Callable | None = None,
    optimizer_options: dict = dict(maxiter=100, ftol=EPS, gtol=0.0),
) -> tuple[Float[Array, "p"], Array | None]:
    """Screen the candidates, then refine the best ones with L-BFGS-B in one dispatch.

    acquisition_loss(c, [extra,] *loss_args) returns (value, gradient); it must be a
    module level function so the compiled refinement is reused across calls.
    extra_args stack per-candidate arguments, loss_args are shared by every candidate.
    screening_loss is an already batched value-only loss, taking every candidate at once.
    """
    candidates = jnp.asarray(candidates)
    extras = (
        None if extra_args is None or len(extra_args) == 0 else jnp.asarray(extra_args)
    )

    # only keep the best initial candidates
    if screening_loss is None:
        losses = [
            (
                acquisition_loss(c, *loss_args)[0]
                if extras is None
                else acquisition_loss(c, e, *loss_args)[0]
            )
            for c, e in zip(candidates, candidates if extras is None else extras)
        ]
    elif extras is None:
        losses = screening_loss(candidates)
    else:
        losses = screening_loss(candidates, extras)
    best = np.argsort(losses)[:max_restarts]
    candidates = candidates[best]
    extras = None if extras is None else extras[best]

    # arrays are traced, everything else keys the compilation cache
    full_args = loss_args if extras is None else (extras, *loss_args)
    dynamic_args, static_args = eqx.partition(full_args, eqx.is_array)
    scalar_loss = autodiff_view(acquisition_loss, static_args)
    in_axes = ((0,) if extras is not None else ()) + (None,) * len(loss_args)

    xs, fs = refine_candidates(
        scalar_loss,
        candidates,
        dynamic_args,
        in_axes,
        optimizer_options["maxiter"],
        optimizer_options["ftol"],
        optimizer_options["gtol"],
    )

    # a restart whose linesearch died on a non finite loss must not win the argmin
    best = jnp.argmin(jnp.where(jnp.isnan(fs), jnp.inf, fs))
    return xs[best], (None if extras is None else extras[best])

def upper_confidence_bound(mu: Scalar, sigma: Scalar, beta: Scalar) -> Scalar:
    return -mu + jnp.sqrt(beta) * sigma

def log_expected_improvement(mu: Scalar, sigma: Scalar, y_best: Scalar) -> Scalar:
    """Stable log EI: log(sigma) + log(pdf(z) + z * cdf(z)) (Ament et al. 2023)."""

    # sanitize inputs for the three branches to avoid NaNs and Infs in gradients
    z = (y_best - mu) / sigma
    upper, lower = z > -1, z < -1 / jnp.sqrt(EPS)

    # branch1 (z > -1): direct evaluation
    z1 = jnp.where(upper, z, 0.0)
    log_h1 = jnp.log(jsp.stats.norm.pdf(z1) + z1 * jsp.stats.norm.cdf(z1))

    # branch2 (-1/sqrt(EPS) <= z <= -1): stable log1mexp trick
    z2 = jnp.where(upper | lower, -2.0, z)
    log_h2 = (
        -(z2**2) / 2
        - jnp.log(2 * jnp.pi) / 2
        + jax.nn.log1mexp(
            -jnp.log(-z2)
            - jsp.stats.norm.logsf(-z2)
            - z2**2 / 2
            - jnp.log(2 * jnp.pi) / 2
        )
    )

    # branch3 (z < -1/sqrt(EPS)): asymptotic expansion
    z3 = jnp.where(lower, z, -2.0 / EPS)
    log_h3 = -(z3**2) / 2 - jnp.log(2 * jnp.pi) / 2 - 2 * jnp.log(-z3)

    log_h = jnp.where(upper, log_h1, jnp.where(lower, log_h3, log_h2))
    return jnp.log(sigma) + log_h



