from typing import Callable
from functools import partial, wraps
from collections import defaultdict
from jaxtyping import Array, Float, Scalar
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import scipy as sp
import numpy as np

import argparse
import time
import os
import pickle

from bofus import gp, kernels, acquisition, rkhs
import targets

jax.config.update("jax_enable_x64", True)

Candidate = rkhs.Function | rkhs.BernsteinPolynomial

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/neurips")

VERBOSE = False

# the pullback metric goes singular as a basis point approaches a lower stratum, and rcond
# truncation drops the directions below a relative singular value outright
LSTSQ_RCOND = None


SAVE_DTYPE = np.float32


def downcast(tree, dtype: np.dtype = SAVE_DTYPE):
    """Store every float array at reduced precision; the pickles run to gigabytes otherwise."""

    def cast(leaf):
        if isinstance(leaf, (jax.Array, np.ndarray)) and leaf.dtype.kind == "f":
            return np.asarray(leaf, dtype)
        return leaf

    return jax.tree.map(cast, tree)


def vprint(*args, **kwargs) -> None:
    """Print only under --verbose; sweeps run thousands of these and the logs add up."""
    if VERBOSE:
        print(*args, **kwargs)


def sample_gp_prior_values(
    kernel: rkhs.RKHS,
    xs: Float[Array, "n d"],
    rng: np.random.Generator,
    size: int = 1,
) -> Float[Array, "size n m"]:
    """Sample every output independently from the prior with covariance kernel(xs, xs)."""
    K_sqrt = jnp.linalg.cholesky(kernel(xs, xs) + 1e-8 * jnp.eye(len(xs)))
    z = rng.standard_normal((size, len(xs), kernel.m))
    return jnp.einsum("np,spm->snm", K_sqrt, z)


def timed(stage: str) -> Callable:
    """Accumulate the wall time of the decorated stage into self.timings[stage]."""

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            vprint(f"{stage.replace('_', ' ').capitalize()}...")
            timer = time.perf_counter()
            result = method(self, *args, **kwargs)
            self.timings[stage] += time.perf_counter() - timer
            vprint(f"Done! (total {stage} time: {self.timings[stage]:.2f}s)\n")
            return result

        return wrapper

    return decorator


# acquisition losses live at module level, so every varying quantity is an argument and
# the jit cache is reused across iterations instead of being rebuilt with each closure


def negative_log_ei(mu: Scalar, cov: Scalar, y_best: Scalar) -> Scalar:
    return -acquisition.log_expected_improvement(
        mu=mu.squeeze(), sigma=cov.squeeze() ** 0.5, y_best=y_best
    )


def basis_points(
    p: Float[Array, "k*(d+m)"], p_dim: int, constrain_order: bool = False
) -> Float[Array, "k d+m"]:
    """Unpack a flat parametrization into (x, y) basis points.

    Under constrain_order the leading input coordinate arrives as k+1 normalized gaps, so
    the chart spans exactly the sorted representative of each permutation orbit.
    """
    if not constrain_order:
        return p.reshape(-1, p_dim)

    k = (len(p) - 1) // p_dim
    gaps, rest = p[: k + 1], p[k + 1 :].reshape(k, p_dim - 1)
    x0 = jnp.cumsum(gaps)[:k] / jnp.maximum(jnp.sum(gaps), acquisition.EPS)
    return jnp.concat([x0[:, None], rest], axis=-1)


def rkhs_ei(
    p: Float[Array, "k*(d+m)"],
    posterior,
    kernel: rkhs.RKHS,
    constrain_order: bool = False,
) -> Scalar:
    """EI of the function parametrized by the flattened (x, y) basis points."""
    f = rkhs.Function.from_array(
        kernel, basis_points(p, kernel.d + kernel.m, constrain_order)
    )
    mu, cov = posterior.predict([f])
    return negative_log_ei(mu, cov, posterior.y_best)


@partial(jax.jacobian, argnums=1)
@partial(jax.jacobian, argnums=0)
def rkhs_preconditioner(
    p1, p2, kernel: rkhs.RKHS, constrain_order: bool = False
) -> Float[Array, "n n"]:
    p_dim = kernel.d + kernel.m
    f1 = rkhs.Function.from_array(kernel, basis_points(p1, p_dim, constrain_order))
    f2 = rkhs.Function.from_array(kernel, basis_points(p2, p_dim, constrain_order))
    return rkhs.inner_product(f1, f2)


@eqx.filter_jit
def rkhs_screening(
    ps, posterior, kernel: rkhs.RKHS, constrain_order: bool = False
) -> Float[Array, "n"]:
    return jax.vmap(rkhs_ei, in_axes=(0, None, None, None))(
        ps, posterior, kernel, constrain_order
    )


def precondition(
    G: Float[Array, "n n"],
    grad: Float[Array, "n"],
    rcond: float | None = LSTSQ_RCOND,
) -> Float[Array, "n"]:
    """Solve G v = grad by pseudoinverse."""
    return jnp.linalg.lstsq(G, grad, rcond=rcond)[0]


@eqx.filter_jit
def rkhs_loss(
    p,
    posterior,
    kernel: rkhs.RKHS,
    natural_gradient: bool = True,
    constrain_order: bool = False,
    rcond: float | None = LSTSQ_RCOND,
):
    val, grad = jax.value_and_grad(rkhs_ei)(p, posterior, kernel, constrain_order)
    if natural_gradient:
        # applying the same chart inside the preconditioner yields the pullback metric
        G = rkhs_preconditioner(p, p, kernel, constrain_order)
        grad = precondition(G, grad, rcond)
    return val, grad


def adaptive_function(p: Float[Array, "k*(d+m)+d"], d: int, m: int) -> rkhs.Function:
    """Decode a candidate that carries its own lengthscale in the trailing d entries."""
    # the lengthscale is optimized in log scale
    log_lo, log_hi = np.log(rkhs.RHO_RANGE[0]), np.log(rkhs.RHO_RANGE[1])
    rho = jnp.exp(p[-d:] * (log_hi - log_lo) + log_lo)
    kernel = rkhs.RKHS(kernels.Euclidean(), kernels.SquaredExponential(), rho, m)
    return rkhs.Function.from_array(kernel, p[:-d].reshape(-1, d + m))


def adaptive_ei(p: Float[Array, "k*(d+m)+d"], posterior, d: int, m: int) -> Scalar:
    """EI of the function parametrized by basis points plus its own lengthscale."""
    mu, cov = posterior.predict([adaptive_function(p, d, m)])
    return negative_log_ei(mu, cov, posterior.y_best)


@partial(jax.jacobian, argnums=1)
@partial(jax.jacobian, argnums=0)
def adaptive_preconditioner(
    p1, p2, ambient: rkhs.RKHS, d: int, m: int
) -> Float[Array, "n n"]:
    f1, f2 = adaptive_function(p1, d, m), adaptive_function(p2, d, m)
    return rkhs.ambient_inner_product(ambient.rho, f1, f2)


@eqx.filter_jit
def adaptive_screening(ps, posterior, d: int, m: int) -> Float[Array, "n"]:
    return jax.vmap(adaptive_ei, in_axes=(0, None, None, None))(ps, posterior, d, m)


@eqx.filter_jit
def adaptive_loss(
    p,
    posterior,
    ambient: rkhs.RKHS,
    d: int,
    m: int,
    natural_gradient: bool = True,
    rcond: float | None = LSTSQ_RCOND,
):
    val, grad = jax.value_and_grad(adaptive_ei)(p, posterior, d, m)
    if natural_gradient:
        # applying the same chart inside the preconditioner yields the pullback metric
        G = adaptive_preconditioner(p, p, ambient, d, m)
        grad = precondition(G, grad, rcond)
    return val, grad


def grid_ei(
    y: Float[Array, "k*m"], x: Float[Array, "k d"], posterior, kernel: rkhs.RKHS
) -> Scalar:
    """EI of the function interpolating the values y (in [0, 1]) on the grid x."""
    y = 2 * y - 1  # [0, 1] -> [-1, 1]
    f = rkhs.Function.from_xy(kernel, x=x, y=y.reshape(len(x), -1))
    mu, cov = posterior.predict([f])
    return negative_log_ei(mu, cov, posterior.y_best)


@partial(jax.jacobian, argnums=1)
@partial(jax.jacobian, argnums=0)
def grid_preconditioner(y1, y2, x, kernel: rkhs.RKHS) -> Float[Array, "n n"]:
    y1 = 2 * y1 - 1  # [0, 1] -> [-1, 1]
    y2 = 2 * y2 - 1  # [0, 1] -> [-1, 1]
    f1 = rkhs.Function.from_xy(kernel, x=x, y=y1.reshape(len(x), -1))
    f2 = rkhs.Function.from_xy(kernel, x=x, y=y2.reshape(len(x), -1))
    return rkhs.inner_product(f1, f2)


@eqx.filter_jit
def grid_screening(ys, xs, posterior, kernel: rkhs.RKHS) -> Float[Array, "n"]:
    return jax.vmap(grid_ei, in_axes=(0, 0, None, None))(ys, xs, posterior, kernel)


@eqx.filter_jit
def grid_loss(
    y,
    x,
    posterior,
    kernel: rkhs.RKHS,
    natural_gradient: bool = True,
    rcond: float | None = LSTSQ_RCOND,
):
    val, grad = jax.value_and_grad(grid_ei)(y, x, posterior, kernel)
    if natural_gradient:
        G = grid_preconditioner(y, y, x, kernel)
        grad = precondition(G, grad, rcond)
    return val, grad


@eqx.filter_jit
def subspace_combination(
    basis_fs: list[rkhs.Function],
    coefficients: Float[Array, "n"],
    kernel: rkhs.RKHS,
) -> rkhs.Function:
    """Squashed linear combination of the basis functions, sharing all their basis points."""
    coefficients = 2 * coefficients - 1  # [0, 1] -> [-1, 1]
    x = [fi.x for fi in basis_fs]
    a = [fi.a * ci for fi, ci in zip(basis_fs, coefficients)]
    f = rkhs.Function(kernel, x=jnp.concat(x), a=jnp.concat(a))
    y = jax.nn.tanh(jax.vmap(f)(f.x))  # squash to [-1, 1]
    return rkhs.Function.from_xy(kernel, x=f.x, y=y)


def subspace_ei(c, basis_fs, posterior, kernel: rkhs.RKHS) -> Scalar:
    f = subspace_combination(basis_fs, c, kernel)
    mu, cov = posterior.predict([f])
    return negative_log_ei(mu, cov, posterior.y_best)


@eqx.filter_jit
def subspace_screening(cs, basis_fs, posterior, kernel: rkhs.RKHS) -> Float[Array, "n"]:
    return jax.vmap(subspace_ei, in_axes=(0, None, None, None))(
        cs, basis_fs, posterior, kernel
    )


@eqx.filter_jit
def subspace_loss(c, basis_fs, posterior, kernel: rkhs.RKHS):
    return jax.value_and_grad(subspace_ei)(c, basis_fs, posterior, kernel)


@eqx.filter_jit
def prior_combination(
    xs_grid: Float[Array, "n d"],
    ys_grids: Float[Array, "b n m"],
    b_grid: Float[Array, "n m"],
    coefficients: Float[Array, "b"],
    kernel: rkhs.RKHS,
) -> rkhs.Function:
    """Squashed linear combination of the prior samples, offset by the incumbent."""
    coefficients = 2 * coefficients - 1  # [0, 1] -> [-1, 1]
    ys_grid = jnp.einsum("bnm,b->nm", ys_grids, coefficients) + b_grid
    ys_grid = jax.nn.tanh(ys_grid)  # squash to [-1, 1]
    return rkhs.Function.from_xy(kernel, x=xs_grid, y=ys_grid)


def prior_ei(c, xs_grid, ys_grids, b_grid, posterior, kernel: rkhs.RKHS) -> Scalar:
    f = prior_combination(xs_grid, ys_grids, b_grid, c, kernel)
    mu, cov = posterior.predict([f])
    return negative_log_ei(mu, cov, posterior.y_best)


@eqx.filter_jit
def prior_screening(cs, xs_grid, ys_grids, b_grid, posterior, kernel: rkhs.RKHS):
    return jax.vmap(prior_ei, in_axes=(0, None, None, None, None, None))(
        cs, xs_grid, ys_grids, b_grid, posterior, kernel
    )


@eqx.filter_jit
def prior_loss(c, xs_grid, ys_grids, b_grid, posterior, kernel: rkhs.RKHS):
    return jax.value_and_grad(prior_ei)(c, xs_grid, ys_grids, b_grid, posterior, kernel)


def vector_ei(c: Float[Array, "n+1"], surrogate_model: gp.GaussianProcess) -> Scalar:
    """EI of a plain coefficient vector, under a vanilla GP surrogate."""
    mu, cov = surrogate_model.predict(c[None, :])
    return negative_log_ei(mu, cov, surrogate_model.observed_ys.min())


@eqx.filter_jit
def vector_screening(cs, surrogate_model) -> Float[Array, "n"]:
    return jax.vmap(vector_ei, in_axes=(0, None))(cs, surrogate_model)


@eqx.filter_jit
def vector_loss(c, surrogate_model):
    return jax.value_and_grad(vector_ei)(c, surrogate_model)


def sort_basis_points(
    ps: Float[Array, "n k*(d+m)"], p_dim: int
) -> Float[Array, "n k*(d+m)"]:
    """Order each candidate's basis points by their first input coordinate."""
    rows = ps.reshape(len(ps), -1, p_dim)
    order = jnp.argsort(rows[..., 0], axis=-1)
    return jnp.take_along_axis(rows, order[..., None], axis=1).reshape(len(ps), -1)


@eqx.filter_jit
def sparsify(f: rkhs.Function, k: int) -> rkhs.Function:
    """Sparsify to k basis points using Kernel Matching Pursuit (Vincent & Bengio 2002)"""
    D = f.kernel(f.x, f.x)
    norm = jnp.linalg.norm(D, axis=0)

    def scan_fn(residual, _):
        gamma = jnp.argmax(jnp.linalg.norm(D @ residual, axis=-1) / norm)
        alpha = (D[gamma] @ residual) / norm[gamma] ** 2
        residual = residual - jnp.outer(D[gamma], alpha)
        return residual, (alpha, gamma)

    residual = jax.vmap(f)(f.x).reshape(len(f.x), -1)
    residual, (a, idx) = jax.lax.scan(scan_fn, residual, None, length=k)
    return f._replace(a=a, x=f.x[idx])


class Runner:
    """Sequential acquisition loop, with the method specific parts left to subclasses."""

    def __init__(
        self,
        seed: int,
        target_fn: targets.TestFunction,
        kernel: rkhs.RKHS,
        surrogate_model,
        # simulation parameters
        initial_acquisitions: int,
        minimum_k: int,
        maximum_k: int,
        acquisitions_each_k: int,
        acquisition_raw_samples: int,
        acquisition_max_restarts: int,
    ):
        self.rng = np.random.default_rng(seed=seed)
        self.target_fn = target_fn
        self.kernel = kernel
        self.surrogate_model = surrogate_model
        self.initial_acquisitions = initial_acquisitions
        self.minimum_k = minimum_k
        self.maximum_k = maximum_k
        self.acquisitions_each_k = acquisitions_each_k
        self.acquisition_raw_samples = acquisition_raw_samples
        self.acquisition_max_restarts = acquisition_max_restarts

        self.fs: list[Candidate] = []
        self.ys: Float[Array, "n"] = jnp.zeros(0)
        self.timings: dict[str, float] = defaultdict(float)

    @property
    def p_dim(self) -> int:
        """Size of the flattened (x, y) parametrization of a single basis point."""
        return self.kernel.d + self.kernel.m

    @property
    def posterior(self) -> gp.FunctionalPosterior:
        return self.surrogate_model.posterior

    def lhs(self, d: int) -> sp.stats.qmc.LatinHypercube:
        return sp.stats.qmc.LatinHypercube(d=d, rng=self.rng)

    def to_function(self, p: Float[Array, "k*(d+m)"], k: int) -> rkhs.Function:
        return rkhs.Function.from_array(self.kernel, p.reshape(k, self.p_dim))

    # method specific hooks
    def initial_candidates(self) -> list[Candidate]:
        raise NotImplementedError

    def propose(self, k: int) -> list[Candidate]:
        raise NotImplementedError

    def on_new_k(self, k: int) -> None:
        """Rebuild whatever depends on k, before the acquisitions at that k start."""

    def fit(self) -> None:
        # observations are append-only, so the previous distance block stays valid
        self.surrogate_model = self.surrogate_model.fit(
            self.fs, self.ys, cached_dists=self.surrogate_model.d2
        )

    def steps_per_k(self, k: int) -> int:
        return self.acquisitions_each_k

    # timed stages
    @timed("acquisition")
    def _acquire(self, k: int) -> list[Candidate]:
        return self.propose(k)

    @timed("acquisition")
    def _acquire_initial(self) -> list[Candidate]:
        return self.initial_candidates()

    @timed("target_evaluation")
    def _evaluate(self, fs: list[Candidate]) -> Float[Array, "n"]:
        return jnp.array([self.target_fn(f) for f in fs])

    @timed("surrogate_fit")
    def _fit_surrogate(self) -> None:
        self.fit()

    def _record(self, fs: list[Candidate], ys: Float[Array, "n"]) -> None:
        self.fs = self.fs + list(fs)
        self.ys = jnp.concatenate([self.ys, ys])
        self._fit_surrogate()

    def run(self) -> dict:
        fs = self._acquire_initial()
        self._record(fs, self._evaluate(fs))

        for k in range(self.minimum_k, self.maximum_k + 1):
            self.on_new_k(k)
            for i in range(self.steps_per_k(k)):
                fs = self._acquire(k)
                self._record(fs, self._evaluate(fs))
                vprint(
                    f"Iteration {i+1} (k={k}): "
                    f"current = {self.ys[-1]:.8f}, best = {self.ys.min():.8f}\n"
                )

        return dict(
            observation_locations=self.fs,
            observation_values=self.ys,
            **{f"{stage}_time": t for stage, t in self.timings.items()},
        )


class Random(Runner):
    """Latin hypercube samples, no surrogate model."""

    def initial_candidates(self) -> list[rkhs.Function]:
        sampler = self.lhs(self.minimum_k * self.p_dim)
        return [
            self.to_function(p, self.minimum_k)
            for p in sampler.random(n=self.initial_acquisitions)
        ]

    def on_new_k(self, k: int) -> None:
        self.candidate_sampler = self.lhs(k * self.p_dim)

    def propose(self, k: int) -> list[rkhs.Function]:
        ps = self.candidate_sampler.random(n=self.acquisitions_each_k)
        return [self.to_function(p, k) for p in ps]

    def steps_per_k(self, k: int) -> int:
        return 1  # the whole batch is drawn at once

    def fit(self) -> None:
        pass


class Ours(Runner):
    """Optimize the acquisition function over the RKHS parametrization itself."""

    def __init__(
        self,
        *args,
        use_natural_gradient: bool = True,
        lstsq_rcond: float | None = LSTSQ_RCOND,
        sample_candidates_from_gp: bool = False,
        constrain_order: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_natural_gradient = use_natural_gradient
        self.lstsq_rcond = lstsq_rcond
        self.sample_candidates_from_gp = sample_candidates_from_gp
        self.constrain_order = constrain_order
        self.grid_sampler = self.lhs(self.kernel.d)

    def as_gaps(self, ps: Float[Array, "n k*(d+m)"]) -> Float[Array, "n k*(d+m)+1"]:
        """Re-encode the leading input coordinate of each basis point as normalized gaps.

        Successive differences of the sorted coordinate sum to one, so basis_points inverts
        this exactly and the pool is the unconstrained one relabelled, not a different draw.
        """
        rows = sort_basis_points(ps, self.p_dim).reshape(len(ps), -1, self.p_dim)
        gaps = jnp.diff(rows[..., 0], axis=-1, prepend=0.0, append=1.0)
        return jnp.concat([gaps, rows[..., 1:].reshape(len(ps), -1)], axis=-1)

    def sample_candidate_from_gp(self, k: int) -> Float[Array, "k*(d+m)"]:
        """Draw the (x, y) parametrization of a single candidate from the GP prior."""
        xs = self.grid_sampler.random(n=k)
        ys = sample_gp_prior_values(self.kernel, xs, self.rng)[0]
        ys = jsp.special.expit(2 * ys)  # squash to [0, 1]
        return jnp.concat([xs, ys], axis=-1).flatten()

    def sample_candidates(self, k: int, n: int) -> Float[Array, "n k*(d+m)"]:
        if self.sample_candidates_from_gp:
            ps = jnp.array([self.sample_candidate_from_gp(k) for _ in range(n)])
        else:
            ps = jnp.array(self.candidate_sampler.random(n=n))
        return self.as_gaps(ps) if self.constrain_order else ps

    def to_function(self, p: Float[Array, "k*(d+m)"], k: int) -> rkhs.Function:
        return rkhs.Function.from_array(
            self.kernel, basis_points(p, self.p_dim, self.constrain_order)
        )

    def initial_candidates(self) -> list[rkhs.Function]:
        self.on_new_k(self.minimum_k)
        ps = self.sample_candidates(self.minimum_k, self.initial_acquisitions)
        return [self.to_function(p, self.minimum_k) for p in ps]

    def on_new_k(self, k: int) -> None:
        self.candidate_sampler = self.lhs(k * self.p_dim)

    def propose(self, k: int) -> list[rkhs.Function]:
        ps = self.sample_candidates(k, self.acquisition_raw_samples)
        posterior, kernel = self.posterior, self.kernel
        p, _ = acquisition.optimize_lhs_candidates(
            acquisition_loss=rkhs_loss,
            loss_args=(
                posterior,
                kernel,
                self.use_natural_gradient,
                self.constrain_order,
                self.lstsq_rcond,
            ),
            candidates=ps.reshape(len(ps), -1),
            max_restarts=self.acquisition_max_restarts,
            screening_loss=lambda ps: rkhs_screening(
                ps, posterior, kernel, self.constrain_order
            ),
        )
        return [self.to_function(p, k)]


class OursAdaptive(Runner):
    """Ours, with each candidate's lengthscale searched along with its basis points."""

    def __init__(
        self,
        *args,
        use_natural_gradient: bool = True,
        lstsq_rcond: float | None = LSTSQ_RCOND,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_natural_gradient = use_natural_gradient
        self.lstsq_rcond = lstsq_rcond

    def to_function(self, p: Float[Array, "k*(d+m)+d"], k: int) -> rkhs.Function:
        return adaptive_function(p, self.kernel.d, self.kernel.m)

    def initial_candidates(self) -> list[rkhs.Function]:
        self.on_new_k(self.minimum_k)
        ps = self.candidate_sampler.random(n=self.initial_acquisitions)
        return [self.to_function(p, self.minimum_k) for p in ps]

    def on_new_k(self, k: int) -> None:
        # the trailing d entries carry the candidate's own lengthscale
        self.candidate_sampler = self.lhs(k * self.p_dim + self.kernel.d)

    def propose(self, k: int) -> list[rkhs.Function]:
        ps = jnp.array(self.candidate_sampler.random(n=self.acquisition_raw_samples))
        posterior, ambient = self.posterior, self.surrogate_model.ambient
        d, m = self.kernel.d, self.kernel.m
        p, _ = acquisition.optimize_lhs_candidates(
            acquisition_loss=adaptive_loss,
            loss_args=(
                posterior,
                ambient,
                d,
                m,
                self.use_natural_gradient,
                self.lstsq_rcond,
            ),
            candidates=ps,
            max_restarts=self.acquisition_max_restarts,
            screening_loss=lambda ps: adaptive_screening(ps, posterior, d, m),
        )
        return [self.to_function(p, k)]


class Vellanky(Runner):
    """Bernstein polynomial coefficients, searched as a plain vector space."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.kernel.d == 1, "Vellanky's method only supports 1D input spaces"

    def to_polynomial(self, c: Float[Array, "(n+1)*m"]) -> rkhs.BernsteinPolynomial:
        return rkhs.BernsteinPolynomial.from_array(
            c.reshape(self.kernel.m, -1).squeeze()
        )

    @property
    def coefficients(self) -> Float[Array, "n (deg+1)*m"]:
        """Observed polynomials as coefficient vectors, renormalized back to [0, 1]."""
        return jnp.stack([(f.c.flatten() + 1) / 2 for f in self.fs])

    def fit(self) -> None:
        self.surrogate_model = self.surrogate_model.fit(self.coefficients, self.ys)

    def initial_candidates(self) -> list[rkhs.BernsteinPolynomial]:
        sampler = self.lhs((self.minimum_k + 1) * self.kernel.m)
        return [
            self.to_polynomial(c) for c in sampler.random(n=self.initial_acquisitions)
        ]

    def on_new_k(self, degree: int) -> None:
        # lift every observation to the new degree, so the surrogate sees a single shape
        self.fs = [f.as_degree(degree) for f in self.fs]
        self._fit_surrogate()
        self.candidate_sampler = self.lhs((degree + 1) * self.kernel.m)

    def propose(self, degree: int) -> list[rkhs.BernsteinPolynomial]:
        surrogate_model = self.surrogate_model
        c, _ = acquisition.optimize_lhs_candidates(
            acquisition_loss=vector_loss,
            loss_args=(surrogate_model,),
            candidates=self.candidate_sampler.random(n=self.acquisition_raw_samples),
            max_restarts=self.acquisition_max_restarts,
            screening_loss=lambda cs: vector_screening(cs, surrogate_model),
        )
        return [self.to_polynomial(c)]


class Kundu(Runner):
    """Optimize over the coefficients of a randomly resampled finite dimensional subspace."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the basis is padded to the final size, so its shape never changes
        self.basis_sampler = self.lhs(self.maximum_k * self.p_dim)

    def sample_basis(self, n: int) -> list[rkhs.Function]:
        ps = self.basis_sampler.random(n=n)
        return [self.to_function(p, self.maximum_k) for p in ps]

    def initial_candidates(self) -> list[rkhs.Function]:
        return self.sample_basis(self.initial_acquisitions)

    def on_new_k(self, n: int) -> None:
        self.candidate_sampler = self.lhs(n)

    def propose(self, n: int) -> list[rkhs.Function]:
        vprint(f"Sampling random subspace of dimension {n}...")
        basis_fs = self.sample_basis(n)
        posterior, kernel = self.posterior, self.kernel
        c, _ = acquisition.optimize_lhs_candidates(
            acquisition_loss=subspace_loss,
            loss_args=(tuple(basis_fs), posterior, kernel),
            candidates=self.candidate_sampler.random(n=self.acquisition_raw_samples),
            max_restarts=self.acquisition_max_restarts,
            screening_loss=lambda cs: subspace_screening(
                cs, basis_fs, posterior, kernel
            ),
        )
        # the combination spans every basis point at once, so trim it back to the k budget
        return [sparsify(subspace_combination(basis_fs, c, kernel), k=self.maximum_k)]


class Vien(Runner):
    """Optimize over function values on a shared grid, then sparsify back to k basis points."""

    def __init__(
        self,
        *args,
        use_natural_gradient: bool = True,
        lstsq_rcond: float | None = LSTSQ_RCOND,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_natural_gradient = use_natural_gradient
        self.lstsq_rcond = lstsq_rcond

    def initial_candidates(self) -> list[rkhs.Function]:
        self.on_new_k(self.minimum_k)
        ps = self.candidate_sampler.random(n=self.initial_acquisitions)
        return [self.to_function(p, self.minimum_k) for p in ps]

    def on_new_k(self, k: int) -> None:
        self.candidate_sampler = self.lhs(k * self.p_dim)

    def expanded_candidates(self, k: int) -> tuple[list, list]:
        """Candidates re-expressed on their own basis points plus every observed one."""
        ps = self.candidate_sampler.random(n=self.acquisition_raw_samples)
        candidate_fs = [self.to_function(p, k) for p in ps]

        all_xs = jnp.unique(jnp.concat([f.x for f in self.fs]), axis=0)
        zero_block = jnp.zeros((len(all_xs), self.kernel.m))
        a0 = [jnp.concat([fi.a, zero_block]) for fi in candidate_fs]
        x0 = [jnp.concat([fi.x, all_xs]) for fi in candidate_fs]
        y0 = [
            jax.vmap(rkhs.Function(self.kernel, x=xi, a=ai))(xi).reshape(-1)
            for xi, ai in zip(x0, a0)
        ]
        y0 = [jsp.special.expit(2 * yi) for yi in y0]  # squash to [0, 1]
        return y0, x0

    def propose(self, k: int) -> list[rkhs.Function]:
        y0, x0 = self.expanded_candidates(k)
        posterior, kernel = self.posterior, self.kernel
        y, x = acquisition.optimize_lhs_candidates(
            acquisition_loss=grid_loss,
            loss_args=(
                posterior,
                kernel,
                self.use_natural_gradient,
                self.lstsq_rcond,
            ),
            candidates=jnp.array(y0),
            extra_args=x0,
            max_restarts=self.acquisition_max_restarts,
            screening_loss=lambda ys, xs: grid_screening(ys, xs, posterior, kernel),
        )
        y = 2 * y - 1  # [0, 1] -> [-1, 1]
        f = rkhs.Function.from_xy(kernel, x=x, y=y.reshape(len(x), -1))  # type: ignore
        return [sparsify(f, k=k)]


class Shilton(Runner):
    """Optimize over combinations of GP prior samples, centered on the incumbent."""

    def __init__(self, *args, reduced_grid: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = self.maximum_k if reduced_grid else 50
        self.grid_sampler = self.lhs(self.kernel.d)
        # a full grid carries far more basis points than the k budget, so trim it back
        self.sparsify_proposals = not reduced_grid

    def to_observation(self, f: rkhs.Function) -> rkhs.Function:
        return sparsify(f, k=self.maximum_k) if self.sparsify_proposals else f

    def sample_from_gp_prior(self, basis_size: int) -> tuple[Array, Array]:
        xs = self.grid_sampler.random(n=self.grid_size)
        ys = sample_gp_prior_values(self.kernel, xs, self.rng, size=basis_size)
        return xs, ys

    def initial_candidates(self) -> list[rkhs.Function]:
        xs_grid, ys_grids = self.sample_from_gp_prior(self.initial_acquisitions)
        return [
            self.to_observation(
                rkhs.Function.from_xy(self.kernel, x=xs_grid, y=ys_grid)
            )
            for ys_grid in ys_grids
        ]

    def on_new_k(self, n: int) -> None:
        self.candidate_sampler = self.lhs(n)

    def propose(self, n: int) -> list[rkhs.Function]:
        vprint(f"Sampling random subspace of dimension {n}...")
        xs_grid, ys_grids = self.sample_from_gp_prior(basis_size=n)

        # expand around the incumbent
        incumbent = self.fs[jnp.argmin(self.ys)]
        b_grid = jnp.array([incumbent(x) for x in xs_grid])
        b_grid = b_grid.reshape(len(xs_grid), self.kernel.m)

        posterior, kernel = self.posterior, self.kernel
        c, _ = acquisition.optimize_lhs_candidates(
            acquisition_loss=prior_loss,
            loss_args=(xs_grid, ys_grids, b_grid, posterior, kernel),
            candidates=self.candidate_sampler.random(n=self.acquisition_raw_samples),
            max_restarts=self.acquisition_max_restarts,
            screening_loss=lambda cs: prior_screening(
                cs, xs_grid, ys_grids, b_grid, posterior, kernel
            ),
        )
        return [
            self.to_observation(prior_combination(xs_grid, ys_grids, b_grid, c, kernel))
        ]


def main():
    global VERBOSE

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "random",
            "ours",
            "ours_adaptive",
            "vellanky",
            "vien",
            "kundu",
            "shilton",
        ],
    )
    parser.add_argument("--target_fn", choices=list(targets.TARGET_FNS))
    parser.add_argument("--lengthscale", type=float, required=True)
    parser.add_argument(
        "--profile", choices=["rbf", "matern52", "matern32", "matern12"]
    )
    parser.add_argument("--seed", type=int, required=True)
    # simulation parameters
    parser.add_argument("--initial_acquisitions", type=int, default=10)
    parser.add_argument("--minimum_k", type=int, default=1)
    parser.add_argument("--maximum_k", type=int, default=10)
    parser.add_argument("--acquisitions_each_k", type=int, default=10)
    parser.add_argument("--acquisition_raw_samples", type=int, default=1024)
    parser.add_argument("--acquisition_max_restarts", type=int, default=16)
    # ablations flags, only used by some methods
    parser.add_argument("--disable_natural_gradient", action="store_true")
    parser.add_argument("--lstsq_rcond", type=float, default=LSTSQ_RCOND)
    parser.add_argument("--sample_candidates_from_gp", action="store_true")
    parser.add_argument("--reduced_grid", action="store_true")
    parser.add_argument("--constrain_order", action="store_true")
    parser.add_argument("--fixed_k", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose

    # fixed-k ablation: spend the same total evaluations at a single k
    if args.fixed_k is not None:
        args.acquisitions_each_k *= args.maximum_k - args.minimum_k + 1
        args.minimum_k = args.maximum_k = args.fixed_k

    # problem setup
    target_fn = targets.make_target(args.target_fn)
    kernel = rkhs.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=jnp.array([args.lengthscale] * target_fn.d),
        m=getattr(target_fn, "m", 1),
    )

    # simulation setup
    profile = {
        "rbf": kernels.SquaredExponential(),
        "matern12": kernels.Matern(nu=1 / 2),
        "matern32": kernels.Matern(nu=3 / 2),
        "matern52": kernels.Matern(nu=5 / 2),
    }[args.profile]
    # adaptive candidates need an ambient below their range, so that l1 + l2 - l0 > 0
    ambient = rkhs.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=jnp.full(target_fn.d, rkhs.RHO_RANGE[0]),
        m=getattr(target_fn, "m", 1),
    )
    runner_cls, surrogate_model = {
        "random": (Random, None),
        "ours": (Ours, gp.FunctionalGaussianProcess(profile=profile)),
        "ours_adaptive": (
            OursAdaptive,
            gp.FunctionalGaussianProcess(profile=profile, ambient=ambient),
        ),
        "vellanky": (Vellanky, gp.GaussianProcess(profile=profile)),
        "kundu": (Kundu, gp.FunctionalGaussianProcess(profile=profile)),
        "vien": (Vien, gp.FunctionalGaussianProcess(profile=profile)),
        "shilton": (Shilton, gp.FunctionalGaussianProcess(profile=profile)),
    }[args.method]

    # ablation flags
    ablations = {}
    if args.disable_natural_gradient:
        vprint("Disabling natural gradient...")
        ablations["use_natural_gradient"] = False
    if args.lstsq_rcond != LSTSQ_RCOND:
        vprint(f"Truncating the preconditioner below {args.lstsq_rcond} * s_max...")
        ablations["lstsq_rcond"] = args.lstsq_rcond
    if args.sample_candidates_from_gp:
        vprint("Sampling acquisition candidates from GP prior...")
        ablations["sample_candidates_from_gp"] = True
    if args.reduced_grid:
        vprint("Using reduced grid...")
        ablations["reduced_grid"] = True
    if args.constrain_order:
        vprint("Constraining the search to the permutation simplex...")
        ablations["constrain_order"] = True

    # run simulation
    results = runner_cls(
        seed=args.seed,
        target_fn=target_fn,
        kernel=kernel,
        surrogate_model=surrogate_model,
        initial_acquisitions=args.initial_acquisitions,
        minimum_k=args.minimum_k,
        maximum_k=args.maximum_k,
        acquisitions_each_k=args.acquisitions_each_k,
        acquisition_raw_samples=args.acquisition_raw_samples,
        acquisition_max_restarts=args.acquisition_max_restarts,
        **ablations,
    ).run()

    # save results
    save_dir = f"{RESULTS_DIR}/{args.target_fn}/{args.method}"
    if args.disable_natural_gradient:
        save_dir += f"_no_natural_grad"
    if args.sample_candidates_from_gp:
        save_dir += f"_sample_from_gp"
    if args.reduced_grid:
        save_dir += f"_reduced_grid"
    if args.constrain_order:
        save_dir += f"_constrain_order"
    if args.lstsq_rcond != LSTSQ_RCOND:
        save_dir += f"_rcond_{args.lstsq_rcond:.0e}"
    if args.fixed_k is not None:
        save_dir += f"_fixed_k_{args.fixed_k}"
    save_dir += f"/{args.profile}_lengthscale_{args.lengthscale}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/seed_{args.seed}"
    pickle.dump(downcast(results), open(f"{save_path}.pkl", "wb"))
    vprint(f"Results saved to {save_path}.pkl")


if __name__ == "__main__":
    main()
