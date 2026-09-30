from typing import Callable
from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import scipy as sp
import numpy as np

jax.config.update("jax_enable_x64", True)
EPS = float(jnp.sqrt(jnp.finfo(float).eps))


def optimize_lhs_candidates(
    acquisition_loss: Callable,
    candidates: Float[Array, "n d"],
    extra_args: list = [],
    max_restarts: int = 0,
    screening_loss: Callable | None = None,
    optimizer_options: dict = dict(maxiter=100, ftol=EPS, gtol=0.0),
) -> tuple[Float[Array, "d"], list]:
    """Screen the candidates, then refine the best ones with L-BFGS-B.

    screening_loss is an already batched value-only loss, taking every candidate at once.
    """
    # only keep the best initial candidates
    extra_args = extra_args or [None] * len(candidates)
    loss_fn = acquisition_loss
    if screening_loss is None:
        losses = [
            loss_fn(c)[0] if args is None else loss_fn(c, args)[0]
            for c, args in zip(candidates, extra_args)
        ]
    elif extra_args[0] is None:
        losses = screening_loss(jnp.asarray(candidates))
    else:
        losses = screening_loss(jnp.asarray(candidates), jnp.asarray(extra_args))
    best = np.argsort(losses)[:max_restarts]
    candidates, extra_args = candidates[best], [extra_args[i] for i in best]

    # optimize each initial guesses with L-BFGS-B
    results = [
        (
            sp.optimize.minimize(
                fun=loss_fn,
                x0=c,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * len(c),
                options=optimizer_options,
            )
            if args is None
            else sp.optimize.minimize(
                fun=loss_fn,
                x0=c,
                args=args,
                jac=True,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * len(c),
                options=optimizer_options,
            )
        )
        for c, args in zip(candidates, extra_args)
    ]

    # sort results and return the best one
    losses = jnp.array([result.fun for result in results])
    best_candidate = results[jnp.argmin(losses)].x
    best_args = extra_args[jnp.argmin(losses)]
    return best_candidate, best_args


def log_expected_improvement(mu: Scalar, sigma: Scalar, y_best: Scalar) -> Scalar:
    # numerically stable version following https://arxiv.org/pdf/2310.20708:
    z = (y_best - mu) / sigma

    # use lax.cond to avoid propagating NaNs in the gradients
    branch1 = lambda: jnp.log(z * jsp.stats.norm.cdf(z) + jsp.stats.norm.pdf(z))
    branch2a = lambda: -2 * jnp.log(-z)
    branch2b = lambda: jax.nn.log1mexp(
        -jnp.log(-z) - jsp.stats.norm.logsf(-z) - z**2 / 2 - jnp.log(2 * jnp.pi) / 2.0
    )
    branch2 = lambda: (
        -(z**2) / 2
        - jnp.log(2 * jnp.pi) / 2
        + jax.lax.cond(z < -1 / jnp.sqrt(EPS), branch2a, branch2b)
    )
    ei = jnp.log(sigma) + jax.lax.cond(z > -1, branch1, branch2)
    return ei


def upper_confidence_bound(mu: Scalar, sigma: Scalar, beta: Scalar) -> Scalar:
    return -mu + jnp.sqrt(beta) * sigma
