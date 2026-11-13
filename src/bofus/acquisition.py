from jaxtyping import Array, Bool, Float, Int, Key, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import equinox as eqx
import vlse.optim

from . import rkhs, gp, utils

A_MAX = 3.0  # amplitude box [-A_MAX, A_MAX], wide enough for tanh to saturate


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
    """Stable log EI: log(sigma) + log(pdf(z) + z * cdf(z)) (Ament et al. 2023).
    https://arxiv.org/abs/2310.20708
    """

    # sanitize inputs for the three branches to avoid NaNs and Infs in gradients
    z = (y_best - mu) / sigma
    z_asymptotic = -jnp.finfo(z.dtype).eps ** -0.25
    upper, lower = z > -1, z < z_asymptotic

    # branch1 (z > -1): direct evaluation
    z1 = jnp.where(upper, z, 0.0)
    log_h1 = jnp.log(jsp.stats.norm.pdf(z1) + z1 * jsp.stats.norm.cdf(z1))

    # branch2 (z_asymptotic <= z <= -1): stable log1mexp trick
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

    # branch3 (z < z_asymptotic): asymptotic expansion
    z3 = jnp.where(lower, z, 2.0 * z_asymptotic)
    log_h3 = -(z3**2) / 2 - jnp.log(2 * jnp.pi) / 2 - 2 * jnp.log(-z3)

    log_h = jnp.where(upper, log_h1, jnp.where(lower, log_h3, log_h2))
    return jnp.log(sigma) + log_h


@jax.jit
def q_log_expected_improvement(
    mu: Float[Array, "... q"],
    cov: Float[Array, "... q q"],
    y_best: Float[Array, "..."],
    eps: Float[Array, "s q"],
    tau_relu: Scalar | float = 1e-6,
    tau_max: Scalar | float = 1e-2,
) -> Float[Array, "..."]:
    """Stable Monte Carlo batch log EI with fat-tailed relu and smooth max (Ament et al. 2023).
    https://arxiv.org/abs/2310.20708
    """
    L = jnp.linalg.cholesky(cov)
    ys = mu[..., None, :] + jnp.einsum("sq,...rq->...sr", eps, L)

    # log softplus(y_best - y) at temperature tau_relu, branches sanitized for finite gradients
    z = (jnp.asarray(y_best)[..., None, None] - ys) / tau_relu
    z_mid, z_high = z.clip(-30.0, 30.0), jnp.where(z > 30.0, z, 1.0)
    log_softplus = jnp.where(
        z > 30.0,
        jnp.log(z_high),
        jnp.where(z < -30.0, z, jnp.log(jax.nn.softplus(z_mid))),
    )

    # cauchy tail keeps gradients alive for samples far worse than the incumbent
    log_cauchy = jnp.log(0.1) - jnp.log1p(z**2)
    log_imp = jnp.log(tau_relu) + jnp.logaddexp(log_softplus, log_cauchy)

    # smooth max over the batch at temperature tau_max, then log mean over the samples
    log_max = tau_max * jax.nn.logsumexp(log_imp / tau_max, axis=-1)
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
    batch_size: int = 1,
    multi_starts: int = 32,
    n_probes: int = 1024,
    n_mc: int = 512,
    n_best: int = 4,
    sigma_around_best: float = 1e-2,
    eta: float = 2.0,
) -> rkhs.RBFMixture["q o m d"]:
    """Maximize Monte Carlo batch EI over RBF mixtures, screening probe batches then L-BFGS-B."""
    _, o, m, d = surrogate.f.l.shape
    x_range = (jnp.asarray(-rkhs.X_MAX), jnp.asarray(rkhs.X_MAX))
    a_range = (jnp.asarray(-A_MAX), jnp.asarray(A_MAX))
    mask = jnp.isfinite(surrogate.y)

    log_l_range = (jnp.log(l_range[0]), jnp.log(l_range[1]))
    bounds = tuple(zip(log_l_range, x_range, a_range))

    def mixture(log_l, x, a) -> rkhs.RBFMixture["... m d"]:
        return rkhs.RBFMixture(jnp.exp(log_l), x, a)

    def box_sample(key: Key, shape: tuple[int, ...]) -> tuple[Array, Array, Array]:
        """Latin hypercube (log_l, x, a) over the box, shape counts up to the atoms."""
        p = utils.latin_hypercube_sample(key, (*shape, 2 * d + 1))
        log_l, x, a = jnp.split(p, [d, 2 * d], axis=-1)
        return jax.tree.map(utils.rescale, (log_l, x, a.squeeze(-1)), *bounds)

    # incumbent as the best posterior mean over the observations, consistent with the nugget
    mu_obs, _ = surrogate.predict_marginal(surrogate.f)
    y_best = jnp.where(mask, mu_obs, jnp.inf).min()

    # half the probe batches cover the box, half perturb the n_best observed points
    key_cover, key_center, key_fresh, key_perturb, key_select, key_mc = jr.split(key, 6)
    n_cover = n_probes // 2
    n_around = n_probes - n_cover
    cover = box_sample(key_cover, (n_cover, batch_size, o, m))
    order = jnp.argsort(jnp.where(mask, surrogate.y, jnp.inf))
    n_top = jnp.minimum(n_best, mask.sum())
    center = order[jr.randint(key_center, (n_around, batch_size), 0, n_top)]
    log_l, x, a = jnp.log(surrogate.f.l)[center], surrogate.f.x[center], surrogate.f.a[center]

    # inert padded atoms get fresh lengthscales and centers, so the optimizer can grow them
    fresh_log_l, fresh_x, _ = box_sample(key_fresh, a.shape)
    inert = (a == 0.0)[..., None]
    around = (jnp.where(inert, fresh_log_l, log_l), jnp.where(inert, fresh_x, x), a)

    # gaussian perturbation relative to each range, clipped to the box
    perturb = lambda k, z, lo, hi: (
        z + sigma_around_best * (hi - lo) * jr.normal(k, z.shape)
    ).clip(lo, hi)
    around = jax.tree.map(perturb, tuple(jr.split(key_perturb, 3)), around, *bounds)
    probes = jax.tree.map(lambda *zs: jnp.concatenate(zs), cover, around)

    # stratified normal draws, fixed so the Monte Carlo objective is deterministic
    tiny = jnp.finfo(surrogate.y.dtype).eps
    u = utils.latin_hypercube_sample(key_mc, (n_mc, batch_size))
    eps = jsp.special.ndtri(u.clip(tiny, 1.0 - tiny))
    tau_relu = 1e-6 * jnp.sqrt(surrogate.nu)

    # analytic log EI for a single point, MC batch log EI else
    def loss(lxa):
        f = mixture(*lxa)
        mu, cov = surrogate.predict_joint(f)
        if batch_size == 1:
            mu, sigma = mu.squeeze(), cov.squeeze() ** 0.5
            return -log_expected_improvement(mu, sigma, y_best)
        cov = cov + 1e-6 * jnp.trace(cov) / batch_size * jnp.eye(batch_size)
        return -q_log_expected_improvement(mu, cov, y_best, eps, tau_relu)

    # screen the probe batches by the joint acquisition, Boltzmann sample the starts
    scores = -jax.vmap(loss)(probes)
    starts = boltzmann_select(key_select, scores, multi_starts, eta)
    starts = jax.tree.map(lambda z: z[starts], probes)

    # box constrained L-BFGS-B from each start
    solve = lambda lxa: vlse.optim.minimise(loss, lxa, bounds=bounds)
    results = jax.vmap(solve)(starts)

    # return the batch with the best Monte Carlo EI
    all_dead = ~jnp.any(jnp.isfinite(results.f))
    results = eqx.error_if(results, all_dead, "all restarts ended with non finite loss")
    best = jnp.nanargmin(results.f)
    return mixture(*jax.tree.map(lambda z: z[best], results.x))
