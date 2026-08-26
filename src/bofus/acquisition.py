from jaxtyping import Array, Float, Key, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import vlse.optim

from . import rkhs, gp, utils


@jax.jit
def upper_confidence_bound(
    mu: Float[Array, "..."],
    sigma: Float[Array, "..."],
    beta: Float[Array, "..."],
) -> Float[Array, "..."]:
    return -mu + jnp.sqrt(beta) * sigma


@jax.jit
def log_expected_improvement(
    mu: Float[Array, "..."],
    sigma: Float[Array, "..."],
    y_best: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Stable log EI: log(sigma) + log(pdf(z) + z * cdf(z)) (Ament et al. 2023)."""

    # sanitize inputs for the three branches to avoid NaNs and Infs in gradients
    z = (y_best - mu) / sigma
    eps = jnp.sqrt(jnp.finfo(z.dtype).eps)
    upper, lower = z > -1, z < -1 / jnp.sqrt(eps)

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
    z3 = jnp.where(lower, z, -2.0 / eps)
    log_h3 = -(z3**2) / 2 - jnp.log(2 * jnp.pi) / 2 - 2 * jnp.log(-z3)

    log_h = jnp.where(upper, log_h1, jnp.where(lower, log_h3, log_h2))
    return jnp.log(sigma) + log_h


@eqx.filter_jit
def optimize_expected_improvement(
    key: Key,
    surrogate: gp.GaussianProcess,
    l_range: tuple[Scalar, Scalar],
    x_range: tuple[Scalar, Scalar],
    y_range: tuple[Scalar, Scalar],
    multi_starts: int = 128,
    n_probes: int = 1024,
) -> rkhs.RBFMixture:
    """Maximise log EI over RBF mixtures, screening random probes then L-BFGS-B."""
    _, k, m, d = surrogate.x.l.shape
    y_best = jnp.nanmin(surrogate.y)

    # the ambient inner product needs l + l_obs - l0 > 0, so clip the lower end
    l_floor = (surrogate.l0.max() - surrogate.x.l.min()) * 1.01
    l_range = (l_range[0].clip(min=l_floor), l_range[1])
    log_l_range = (jnp.log(l_range[0]), jnp.log(l_range[1]))
    bounds = tuple(zip(log_l_range, x_range, y_range))

    # probe the space via latin hypercube, squared lengthscales log-uniform
    p = utils.latin_hypercube_sample(key, (n_probes, k, m, 2 * d + 1))
    log_l, x, y = jnp.split(p, [d, 2 * d], axis=-1)
    log_l = utils.rescale(log_l, *log_l_range)
    x = utils.rescale(x, *x_range)
    y = utils.rescale(y, *y_range)
    candidates = rkhs.RBFMixture.from_lxy(jnp.exp(log_l), x, y.squeeze(-1))

    # keep the probes with the best log EI as multistart candidates
    # add a mock axis so predict computes marginals only
    candidates = jax.tree.map(lambda z: z[:, None], candidates)
    mu, cov = jax.vmap(surrogate.predict)(candidates)
    mu, std = mu.squeeze(-1), cov.squeeze((-2, -1)) ** 0.5
    log_ei = log_expected_improvement(mu, std, y_best)
    _, best = jax.lax.top_k(log_ei, multi_starts)
    candidates = jax.tree.map(lambda z: z[best], candidates)

    # vmap so each candidate and output computes its own y only
    log_l, x = jnp.log(candidates.l), candidates.x
    y = jax.vmap(jax.vmap(jax.vmap(lambda f: f(f.x))))(candidates)

    # box constrained L-BFGS-B
    def loss(lxy):
        log_l, x, y = lxy
        f = rkhs.RBFMixture.from_lxy(jnp.exp(log_l), x, y)
        mu, cov = surrogate.predict(f)
        mu, sigma = mu.squeeze(), cov.squeeze() ** 0.5
        return -log_expected_improvement(mu, sigma, y_best)

    solve = lambda lxy: vlse.optim.minimise(loss, lxy, bounds=bounds)
    results = jax.vmap(solve)((log_l, x, y))

    # return the candidate with the best log EI
    all_dead = ~jnp.any(jnp.isfinite(results.f))
    results = eqx.error_if(results, all_dead, "all restarts ended with non finite loss")
    best = jnp.nanargmin(results.f)
    log_l, x, y = jax.tree.map(lambda z: z[best, 0], results.x)
    return rkhs.RBFMixture.from_lxy(jnp.exp(log_l), x, y)
