from jaxtyping import Array, Float, Int, Key, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
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


@jax.jit
def q_log_expected_improvement(
    mu: Float[Array, "... q"],
    cov: Float[Array, "... q q"],
    y_best: Float[Array, "..."],
    eps: Float[Array, "s q"],
    tau: float = 1e-2,
) -> Float[Array, "..."]:
    """Stable Monte Carlo batch log EI with smoothed relu and max (Ament et al. 2023)."""
    L = jnp.linalg.cholesky(cov)
    ys = mu[..., None, :] + jnp.einsum("sq,...rq->...sr", eps, L)

    # log softplus_tau(y_best - y): smooth per-point log improvement
    # sanitize inputs for the three branches to avoid NaNs and Infs in gradients
    z = (jnp.asarray(y_best)[..., None, None] - ys) / tau
    z_mid, z_high = z.clip(-30.0, 30.0), jnp.where(z > 30.0, z, 1.0)
    log_softplus = jnp.where(
        z > 30.0,
        jnp.log(z_high),
        jnp.where(z < -30.0, z, jnp.log(jax.nn.softplus(z_mid))),
    )
    log_imp = jnp.log(tau) + log_softplus

    # smooth max over the batch, then average the samples in log space
    log_max = tau * jax.nn.logsumexp(log_imp / tau, axis=-1)
    return jax.nn.logsumexp(log_max, axis=-1) - jnp.log(len(eps))


@eqx.filter_jit
def boltzmann_select(
    key: Key,
    values: Float[Array, "n"],
    n: int,
    eta: float = 2.0,
) -> Int[Array, "n"]:
    """Sample n distinct indices with weights exp(eta * standardized values), keeping the argmax."""
    finite = jnp.isfinite(values)
    v = jnp.where(finite, values, jnp.nan)
    z = (v - jnp.nanmean(v)) / jnp.maximum(jnp.nanstd(v), jnp.finfo(v.dtype).eps)
    logits = jnp.where(finite, eta * z, -jnp.inf).at[jnp.nanargmax(v)].set(jnp.inf)

    # gumbel top-k is sampling without replacement
    _, idx = jax.lax.top_k(logits + jr.gumbel(key, values.shape), n)
    return idx


@eqx.filter_jit
def optimize_expected_improvement(
    key: Key,
    surrogate: gp.GaussianProcess,
    l_range: tuple[Scalar, Scalar],
    x_range: tuple[Scalar, Scalar],
    a_range: tuple[Scalar, Scalar],
    batch_size: int = 1,
    multi_starts: int = 32,
    n_probes: int = 1024,
    n_mc: int = 128,
    n_best: int = 4,
    sigma_around_best: float = 1e-2,
    eta: float = 2.0,
) -> rkhs.RBFMixture:
    """Maximise Monte Carlo batch EI over RBF mixtures, screening probes then L-BFGS-B.

    Half the probes cover the box by latin hypercube, half perturb the n_best observed
    points by sigma_around_best in unit-cube coordinates. Starts are Boltzmann
    sampled from marginal log EI at temperature eta, the incumbent is the best
    posterior mean over the observations.
    """
    _, k, m, d = surrogate.x.l.shape
    mask = jnp.isfinite(surrogate.y)

    # ambient inner products need l + l_obs - l0 > 0 and 2l - l0 > 0, so clip the lower end
    l_floor = jnp.maximum(surrogate.l0.max() - surrogate.x.l.min(), surrogate.l0.max() / 2) * 1.01
    l_range = (l_range[0].clip(min=l_floor), l_range[1])
    log_l_range = (jnp.log(l_range[0]), jnp.log(l_range[1]))
    bounds = tuple(zip(log_l_range, x_range, a_range))

    def marginals(f: rkhs.RBFMixture) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
        """Posterior mean and std of each function on its own."""
        f = jax.tree.map(lambda z: z[:, None], f)
        mu, cov = jax.vmap(surrogate.predict)(f)
        return mu.squeeze(-1), cov.squeeze((-2, -1)) ** 0.5

    def to_unit(f: rkhs.RBFMixture) -> Float[Array, "n k m p"]:
        unit = lambda z, lo, hi: (z - lo) / (hi - lo)
        return jnp.concatenate(
            [
                unit(jnp.log(f.l), *log_l_range),
                unit(f.x, *x_range),
                unit(f.a, *a_range)[..., None],
            ],
            axis=-1,
        )

    def from_unit(p: Float[Array, "n k m p"]) -> rkhs.RBFMixture:
        log_l, x, a = jnp.split(p, [d, 2 * d], axis=-1)
        log_l = utils.rescale(log_l, *log_l_range)
        x = utils.rescale(x, *x_range)
        a = utils.rescale(a, *a_range)
        return rkhs.RBFMixture(l=jnp.exp(log_l), x=x, a=a.squeeze(-1))

    # incumbent as the best posterior mean over the observations, consistent with the nugget
    mu_obs, _ = marginals(surrogate.x)
    y_best = jnp.where(mask, mu_obs, jnp.inf).min()

    # half the probes cover the box, half perturb the n_best observed points
    key_cover, key_center, key_perturb, key_select, key_mc = jr.split(key, 5)
    n_cover = n_probes // 2
    n_around = n_probes - n_cover
    p_cover = utils.latin_hypercube_sample(key_cover, (n_cover, k, m, 2 * d + 1))
    order = jnp.argsort(jnp.where(mask, surrogate.y, jnp.inf))
    n_top = jnp.minimum(n_best, mask.sum())
    center = order[jr.randint(key_center, (n_around,), 0, n_top)]
    noise = sigma_around_best * jr.normal(key_perturb, (n_around, k, m, 2 * d + 1))
    p_around = (to_unit(surrogate.x)[center] + noise).clip(0.0, 1.0)
    candidates = from_unit(jnp.concatenate([p_cover, p_around]))

    # screen probes by marginal log EI, Boltzmann sample the starts, stride into batches
    log_ei = log_expected_improvement(*marginals(candidates), y_best)
    starts = boltzmann_select(key_select, log_ei, multi_starts * batch_size, eta)
    starts = starts.reshape(batch_size, multi_starts).T
    candidates = jax.tree.map(lambda z: z[starts], candidates)

    log_l, x, a = jnp.log(candidates.l), candidates.x, candidates.a

    # box constrained L-BFGS-B, analytic log EI for a single point, MC batch log EI else
    eps = jr.normal(key_mc, (n_mc, batch_size))

    def loss(lxa):
        log_l, x, a = lxa
        f = rkhs.RBFMixture(l=jnp.exp(log_l), x=x, a=a)
        mu, cov = surrogate.predict(f)
        if batch_size == 1:
            mu, sigma = mu.squeeze(), cov.squeeze() ** 0.5
            return -log_expected_improvement(mu, sigma, y_best)
        cov = cov + 1e-6 * jnp.trace(cov) / batch_size * jnp.eye(batch_size)
        return -q_log_expected_improvement(mu, cov, y_best, eps)

    solve = lambda lxa: vlse.optim.minimise(loss, lxa, bounds=bounds)
    results = jax.vmap(solve)((log_l, x, a))

    # return the batch with the best Monte Carlo EI
    all_dead = ~jnp.any(jnp.isfinite(results.f))
    results = eqx.error_if(results, all_dead, "all restarts ended with non finite loss")
    best = jnp.nanargmin(results.f)
    log_l, x, a = jax.tree.map(lambda z: z[best], results.x)
    return rkhs.RBFMixture(l=jnp.exp(log_l), x=x, a=a)
