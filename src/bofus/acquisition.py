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


def student_t_cdf(
    z: Float[Array, "..."],
    dof: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Student-t cdf, linearized near z = 0 (cubic error < eps) where betainc's jvp is 0 * inf."""
    EPS = utils.eps(z.dtype)
    near_zero = jnp.abs(z) < EPS ** (1 / 3)
    z2 = jnp.where(near_zero, 1.0, z**2)
    tail = 0.5 * jsp.special.betainc(dof / 2, 0.5, dof / (dof + z2))
    cdf = jnp.where(z < 0, tail, 1 - tail)
    return jnp.where(near_zero, 0.5 + jsp.stats.t.pdf(0.0, dof) * z, cdf)


@jax.jit
def log_student_t_expected_improvement(
    mu: Float[Array, "..."],
    sigma: Float[Array, "..."],
    y_best: Float[Array, "..."],
    dof: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Stable log EI under a student-t, log1mexp in the tail is safe since the ratio stays < (dof-1)/dof."""
    z = (y_best - mu) / sigma
    upper = z > -1
    log_A_pdf = jnp.log((dof + z**2) / (dof - 1)) + jsp.stats.t.logpdf(z, dof)

    # z > -1: direct
    z1 = jnp.where(upper, z, 0.0)
    log_A_pdf1 = jnp.where(upper, log_A_pdf, 0.0)
    log_h1 = jnp.log(z1 * student_t_cdf(z1, dof) + jnp.exp(log_A_pdf1))

    # z <= -1: A * pdf - |z| * cdf in log space
    z2 = jnp.where(upper, -2.0, z)
    log_A_pdf2 = jnp.where(upper, 0.0, log_A_pdf)
    log_ratio = log_A_pdf2 - jnp.log(-z2) - jnp.log(student_t_cdf(z2, dof))
    log_h2 = log_A_pdf2 + jax.nn.log1mexp(log_ratio)

    return jnp.log(sigma) + jnp.where(upper, log_h1, log_h2)


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
    EPS = utils.eps(z.dtype)
    z_asymptotic = -(EPS**-0.25)
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
    EPS = utils.eps(values.dtype)
    finite = jnp.isfinite(values)
    v = jnp.where(finite, values, jnp.nan)
    z = (v - jnp.nanmean(v)) / jnp.maximum(jnp.nanstd(v), EPS)
    logits = jnp.where(finite, eta * z, -jnp.inf).at[jnp.nanargmax(v)].set(jnp.inf)

    # gumbel top-k is sampling without replacement
    _, idx = jax.lax.top_k(logits + jr.gumbel(key, values.shape), n)
    return idx


@eqx.filter_jit
def optimize_expected_improvement(
    key: Key,
    surrogate: gp.GaussianProcess,
    l_range: rkhs.LRange,
    a_range: rkhs.ARange,
    batch_size: int = 1,
    multi_starts: int = 32,
    n_probes: int = 1024,
    probe_chunk: int = 1,
    solve_chunk: int = 1,
    gtol: float = 1e-5,
    ftol: float = 1e-5,
    n_mc: int = 512,
    n_best: int = 4,
    sigma_around_best: float = 1e-2,
    eta: float = 2.0,
) -> rkhs.RBFMixture["q o m d"]:
    """Maximize Monte Carlo batch EI over RBF mixtures, screening probe batches then L-BFGS-B."""
    _, o, m, d = surrogate.f.l.shape
    dtype = surrogate.y.dtype
    EPS = utils.eps(dtype)
    l_range, a_range, x_range = (
        tuple(jnp.asarray(z, dtype) for z in r)
        for r in (l_range, a_range, surrogate.x_range)
    )
    mask = jnp.isfinite(surrogate.y)

    # the search runs over (log_l, x, a), each within its box
    lo = (jnp.log(l_range[0]), x_range[0], a_range[0])
    hi = (jnp.log(l_range[1]), x_range[1], a_range[1])

    def mixture(log_l, x, a) -> rkhs.RBFMixture["... m d"]:
        return rkhs.RBFMixture(jnp.exp(log_l), x, a)

    # incumbent as the best posterior mean over the observations, consistent with the nugget
    mu_obs, _ = surrogate.predict_marginal(surrogate.f)
    y_best = jnp.where(mask, mu_obs, jnp.inf).min()

    # half the probe batches draw from the prior
    key_cover, key_center, key_perturb, key_select, key_mc = jr.split(key, 5)
    n_cover = n_probes // 2
    n_around = n_probes - n_cover
    cover = rkhs.RBFMixture.sample(
        key_cover, (n_cover, batch_size, o, m, d), l_range, a_range, x_range
    )
    cover = (jnp.log(cover.l), cover.x, cover.a)

    # the other half start from the n_best observed points
    order = jnp.argsort(jnp.where(mask, surrogate.y, jnp.inf))
    n_top = jnp.minimum(n_best, mask.sum())
    center = order[jr.randint(key_center, (n_around, batch_size), 0, n_top)]
    around = jax.tree.map(lambda z: z[center], surrogate.f)
    around = (jnp.log(around.l), around.x, around.a)

    # gaussian perturbation of (log_l, x, a) relative to each range, clipped to the box
    def perturb(k, z, z_lo, z_hi):
        noise = sigma_around_best * (z_hi - z_lo) * jr.normal(k, z.shape)
        return (z + noise).clip(z_lo, z_hi)

    around = jax.tree.map(perturb, tuple(jr.split(key_perturb, 3)), around, lo, hi)
    probes = jax.tree.map(lambda *zs: jnp.concatenate(zs), cover, around)

    # stratified normal draws, fixed so the Monte Carlo objective is deterministic
    u = utils.latin_hypercube_sample(key_mc, (n_mc, batch_size))
    eps = jsp.special.ndtri(u.clip(EPS, 1.0 - EPS))

    # student-t draws share one chi-square scale per sample across the batch
    if surrogate.dof is not None:
        chi2 = jr.chisquare(jr.fold_in(key_mc, 1), surrogate.dof, (n_mc,))
        eps = eps * jnp.sqrt(surrogate.dof / chi2)[:, None]

    # relu temperature relative to the posterior scale
    tau_relu = 1e-6 * jnp.sqrt(surrogate.nu)

    # analytic log EI for a single point, MC batch log EI else
    def loss(lxa):
        f = mixture(*lxa)
        mu, cov = surrogate.predict_joint(f)
        if batch_size == 1:
            mu, sigma = mu.squeeze(), cov.squeeze() ** 0.5
            if surrogate.dof is None:
                return -log_expected_improvement(mu, sigma, y_best)
            return -log_student_t_expected_improvement(mu, sigma, y_best, surrogate.dof)
        cov = cov + 1e-6 * jnp.trace(cov) / batch_size * jnp.eye(batch_size)
        return -q_log_expected_improvement(mu, cov, y_best, eps, tau_relu)

    # screen the probe batches by the joint acquisition in chunks, Boltzmann sample the starts
    scores = -jax.lax.map(loss, probes, batch_size=probe_chunk)
    starts = boltzmann_select(key_select, scores, multi_starts, eta)
    starts = jax.tree.map(lambda z: z[starts], probes)

    # box constrained L-BFGS-B from each start, mapped since vmap runs every start to the slowest
    def solve(lxa):
        return vlse.optim.minimise(loss, lxa, bounds=(lo, hi), tol=gtol, ftol=ftol)

    results = jax.lax.map(solve, starts, batch_size=solve_chunk)

    # return the batch with the best Monte Carlo EI
    all_dead = ~jnp.any(jnp.isfinite(results.f))
    results = eqx.error_if(results, all_dead, "all restarts ended with non finite loss")
    best = jnp.nanargmin(results.f)
    return mixture(*jax.tree.map(lambda z: z[best], results.x))
