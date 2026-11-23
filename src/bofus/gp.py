from typing import NamedTuple, Optional
from jaxtyping import Array, Float, Key, Scalar

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax.scipy.linalg import cho_solve, cholesky
from vlse import optim

from . import kernels, rkhs, utils


@jax.jit
def factorize(
    K: Float[Array, "n n"],
    y: Float[Array, "n"],
) -> tuple[
    Float[Array, "n n"], Float[Array, "n"], Float[Array, "n"], Scalar, Scalar, Scalar
]:
    """Cholesky factor and solves of K, with the trend b, the REML scale nu and its dof.
    A nan observation marks padding, masked to an identity block of the factor.
    """
    EPS, TINY = utils.eps(y.dtype), utils.tiny(y.dtype)

    # mask the covariance matrix and observations
    mask = jnp.isfinite(y)
    dof = jnp.maximum(mask.sum() - 1, 1).astype(y.dtype)
    K = utils.mask_covariance(K, mask)
    y = jnp.where(mask, y, 0.0)

    # one solve for K^-1 @ 1 and K^-1 @ y, the trend b weights the observations by Ki_1
    K_chol = cholesky(K, lower=True)
    Ki_1, Ki_y = cho_solve((K_chol, True), jnp.stack([1.0 * mask, y], axis=-1)).T
    b = jnp.average(y, weights=Ki_1)

    # REML scale nu, one dof spent on b
    Ki_r = Ki_y - b * Ki_1
    nu = jnp.sum(jnp.where(mask, (y - b) * Ki_r, 0.0)) / dof

    # constant observations give nu = 0, floor it at their float resolution
    resolution = EPS * jnp.max(jnp.abs(y))
    nu = jnp.maximum(nu, jnp.maximum(resolution**2, TINY))
    return K_chol, Ki_1, Ki_r, b, nu, dof


@jax.jit
def loglikelihood(
    K: Float[Array, "n n"],
    y: Float[Array, "n"],
    nu: Scalar | None = None,
) -> Scalar:
    """REML likelihood: trend integrated out under a flat prior, scale nu concentrated unless given."""
    K_chol, Ki_1, _, _, nu_hat, dof = factorize(K, y)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_chol))) + jnp.log(Ki_1.sum())
    if nu is None:
        return -0.5 * (dof * jnp.log(nu_hat) + logdetK)

    return -0.5 * (dof * jnp.log(nu) + logdetK + dof * nu_hat / nu)


@eqx.filter_jit
def pairwise_squared_distance(
    l0: Float[Array, "d"] | None,
    x_range: rkhs.XRange,
    f1: rkhs.RBFMixture["n1 o m d"],
    f2: Optional[rkhs.RBFMixture["n2 o m d"]] = None,
    *,
    rho: rkhs.RBFMixture["o w d"],
    periodic: tuple[bool, ...] | None = None,
) -> Float[Array, "n1 n2 o"]:
    """Squared distances between two sets of functions, per output.
    Distances compare the atoms deconvolved by the ambient lengthscale l0 (plain atoms if None),
    weighted over the inputs by the mixture rho of each output.
    """
    cols = f1 if f2 is None else f2
    product = lambda a, b: rkhs.inner(a, b, l0, x_range, rho, periodic)
    row_norms, col_norms = product(f1, f1), product(cols, cols)

    # one row at a time, recomputed in the backward pass, so only one row of temporaries is alive
    cross = jax.lax.map(jax.checkpoint(lambda row: product(row, cols)), f1)
    d2 = row_norms[:, None] + col_norms[None, :] - 2.0 * cross

    # inf - inf gives nan, cancellation can dip below zero
    d2 = jnp.where(jnp.isfinite(d2), jnp.where(d2 > 0.0, d2, 0.0), jnp.inf)

    # floating point arithmetic is not guaranteed to cancel out on the diagonal
    if f2 is None:
        d2 = jnp.where(jnp.eye(len(d2), dtype=bool)[..., None], 0.0, d2)
    return d2


class GaussianProcess(NamedTuple):
    """Fitted GP over functions, compared through their rho-weighted distances."""

    profile: kernels.Profile  # kernel profile, e.g. RBF or Matern
    l0: Float[Array, "d"] | None  # ambient lengthscale, None compares the plain atoms
    x_range: rkhs.XRange  # input box (lo, hi)
    periodic: tuple[bool, ...] | None  # dimensions that wrap, static; None for none

    # fitted parameters
    rho: rkhs.RBFMixture[
        "o w d"
    ]  # input weight per output, fitted amplitudes on a fixed basis
    g: Scalar  # noise nugget
    nu: Scalar  # covariance scale
    dof: Scalar | None  # student-t degrees of freedom
    b: Scalar  # mean trend

    # observations, nan values mark padded entries
    f: rkhs.RBFMixture["n o m d"]
    y: Float[Array, "n"]

    # cached for posterior evaluation, K is the covariance before scaling by nu
    K_chol: Float[Array, "n n"]
    Ki_1: Float[Array, "n"]  # K^-1 @ 1
    Ki_r: Float[Array, "n"]  # K^-1 @ (y - b)

    @staticmethod
    @eqx.filter_jit
    def fit(
        key: Key,
        f: rkhs.RBFMixture["n o m d"],
        y: Float[Array, "n"],
        *,
        y_std: Optional[Float[Array, "n"]] = None,
        profile: kernels.Profile = kernels.matern52,
        x_range: rkhs.XRange,
        l0: Optional[Float[Array, "d"]] = None,
        periodic: tuple[bool, ...] | None = None,
        n_atoms: int = 1,
        rho_range: Float[Array, "2"] = jnp.array([1e-2, 1e2]),
        nugget_range: Float[Array, "2"] = jnp.array([1e-4, 1e0]),
        nu_range: Float[Array, "2"] = jnp.array([1e-2, 1e2]),
        restarts: int = 1,
    ) -> "GaussianProcess":
        """MAP fit of (rho, g) on the REML likelihood, nu integrated out unless y_std is given.
        rho has fitted amplitudes per output on n_atoms shared weight atoms: the box weight if one,
        else drawn from the mixture prior. Distances are linear in them, so each atom's are cached.
        """
        n = len(y)
        _, o, _, d = f.x.shape
        key_basis, key = jr.split(key)
        basis = (
            rkhs.box_weight(x_range, d)
            if n_atoms == 1
            else rkhs.RBFMixture.sample(
                key_basis, (n_atoms, d), (0.0, jnp.inf), (1.0, 1.0), x_range
            )
        )

        # squared distances per basis atom and output, one atom at a time
        mask = jnp.isfinite(y)
        atom_d2 = lambda atom: pairwise_squared_distance(
            l0, x_range, f, rho=jax.tree.map(lambda z: z[None], atom), periodic=periodic
        )
        D = jax.lax.map(atom_d2, basis)

        # center the amplitudes to average inverse distances, split evenly across the atoms
        valid = (mask[:, None] & mask[None, :])[None, ..., None] & jnp.isfinite(D)
        valid = eqx.error_if(valid, ~jnp.any(valid, axis=(1, 2)), "no valid distances")
        a_scale = 1 / jnp.mean(D, axis=(1, 2), where=valid).T / n_atoms

        # known measurement variances, zeroed on padding so nan never reaches the gradients
        noise = None if y_std is None else jnp.where(mask, y_std, 0.0) ** 2

        def covariance(
            a: Float[Array, "o k"], g: Scalar, nu: Scalar | None
        ) -> Float[Array, "n n"]:
            K = profile(utils.safe_sqrt(jnp.einsum("kijo,ok->ij", D, a))) + g * jnp.eye(
                n
            )
            return K if nu is None else K + jnp.diag(noise / nu)

        def unpack(params: dict[str, Float[Array, "..."]]) -> tuple:
            nu = nu_scale * jnp.exp(params["nu"]) if "nu" in params else None
            return a_scale * jnp.exp(params["a"]), jnp.exp(params["g"]), nu

        # center the nu search on the noiseless closed form at the centre of the parameter box
        g_center = jnp.sqrt(jnp.prod(nugget_range))
        *_, nu_scale, _ = factorize(covariance(a_scale, g_center, None), y)

        # log-normal priors on the amplitudes and nugget, box = ±3 std; nu flat
        ranges = dict(
            a=(jnp.log(rho_range), (o, n_atoms)), g=(jnp.log(nugget_range), ())
        )
        if y_std is not None:
            ranges |= dict(nu=(jnp.log(nu_range), ()))
        lo, hi = (
            {k: jnp.broadcast_to(r[i], s) for k, (r, s) in ranges.items()}
            for i in (0, 1)
        )
        priors = ["a", "g"]

        def map_loss(params: dict[str, Float[Array, "..."]]) -> Scalar:
            a, g, nu = unpack(params)
            z = [3 * (2 * params[k] - lo[k] - hi[k]) / (hi[k] - lo[k]) for k in priors]
            neg_log_prior = 0.5 * sum(jnp.sum(zi**2) for zi in z)
            return -loglikelihood(covariance(a, g, nu), y, nu) + neg_log_prior

        # L-BFGS-B from uniform restarts over the box
        keys = dict(zip(ranges, jr.split(key, len(ranges))))
        starts = {
            k: jr.uniform(keys[k], (restarts, *lo[k].shape), minval=lo[k], maxval=hi[k])
            for k in ranges
        }
        solve = lambda start: optim.minimise(map_loss, start, bounds=(lo, hi))
        results = jax.vmap(solve)(starts)

        # extract the optimal parameters
        dead = ~jnp.any(jnp.isfinite(results.f))
        results = eqx.error_if(results, dead, "fit ended with non finite loss")
        best = jnp.nanargmin(results.f)
        a, g, nu = unpack(jax.tree.map(lambda z: z[best], results.x))
        rho = rkhs.RBFMixture(
            jnp.broadcast_to(basis.l, (o, n_atoms, d)),
            jnp.broadcast_to(basis.x, (o, n_atoms, d)),
            a,
        )

        # condition on the observations, caching the solves predict needs
        K_chol, Ki_1, Ki_r, b, nu_hat, dof = factorize(covariance(a, g, nu), y)
        # integrating nu out leaves a student-t with n - 1 dof, a known nu a gaussian
        nu, dof = (nu_hat, dof) if nu is None else (nu, None)
        return GaussianProcess(
            profile, l0, x_range, periodic, rho, g, nu, dof, b, f, y, K_chol, Ki_1, Ki_r
        )

    @eqx.filter_jit
    def predict_joint(
        self, f: rkhs.RBFMixture["q o m d"]
    ) -> tuple[Float[Array, "q"], Float[Array, "q q"]]:
        """Posterior location and scale at f, student-t with self.dof dof (gaussian if None)."""

        def dist(*fs: rkhs.RBFMixture) -> Float[Array, "n1 n2"]:
            d2 = pairwise_squared_distance(
                self.l0, self.x_range, *fs, periodic=self.periodic, rho=self.rho
            )
            # cancellation only approximates the exact zeros of coincident functions
            return utils.safe_sqrt(d2.sum(-1))

        Kqq = self.profile(dist(f))
        Knq = self.profile(dist(self.f, f))

        # a nan observation marks padding, zero its cross covariance
        Knq = Knq * jnp.isfinite(self.y)[:, None]

        # posterior mean and unit-scale covariance
        mean = self.b + Knq.T @ self.Ki_r
        cov = Kqq - Knq.T @ cho_solve((self.K_chol, True), Knq)

        # correction from estimating the trend on the same data
        Kbq = 1 - Knq.T @ self.Ki_1
        cov = cov + jnp.outer(Kbq, Kbq) / self.Ki_1.sum()
        return mean, self.nu * cov

    @eqx.filter_jit
    def predict_marginal(
        self, f: rkhs.RBFMixture["q o m d"]
    ) -> tuple[Float[Array, "q"], Float[Array, "q"]]:
        """Posterior location and squared scale of each function on its own."""
        f = jax.tree.map(lambda z: z[:, None], f)
        mean, cov = jax.vmap(self.predict_joint)(f)
        return mean.squeeze(-1), cov.squeeze((-2, -1))
