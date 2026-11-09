# Ambient inner product with per-basis-point lengthscales

## Setup

All kernels are squared exponentials with diagonal precision. For a lengthscale
vector $\rho \in \mathbb{R}^d_{>0}$ write $\Lambda = \mathrm{diag}(\rho^2)$ and

$$
k_\Lambda(x, y) = \exp\!\Big(-\tfrac12 (x-y)^\top \Lambda^{-1} (x-y)\Big).
$$

The *ambient* space is the RKHS $\mathcal{H}_0$ of $k_{\Lambda_0}$ with a fixed
host lengthscale $\rho_0$. Functions are finite kernel expansions whose atoms
may each carry their own lengthscale:

$$
f(t) = \sum_i a_i\, k_{\Lambda_i}(t, x_i),
\qquad
g(t) = \sum_j b_j\, k_{M_j}(t, y_j).
$$

## Inner product of two Gaussian atoms

Via the Fourier characterization of the RKHS norm,
$\langle u, v \rangle_{\mathcal{H}_0} = (2\pi)^{-d} \int \hat u(\omega)\,
\overline{\hat v(\omega)} / \hat k_{\Lambda_0}(\omega)\, d\omega$, and the
Fourier transform of a Gaussian atom
$\widehat{k_\Lambda(\cdot, x)}(\omega) = (2\pi)^{d/2} |\Lambda|^{1/2}
e^{-\frac12 \omega^\top \Lambda \omega} e^{-i \omega^\top x}$,
the integrand is again Gaussian and integrates in closed form:

$$
\big\langle k_{\Lambda_1}(\cdot, u),\, k_{\Lambda_2}(\cdot, v) \big\rangle_{\mathcal{H}_0}
= \left( \frac{|\Lambda_1|\,|\Lambda_2|}{|\Lambda_0|\,|\Lambda_s|} \right)^{1/2}
\exp\!\Big(-\tfrac12 (u-v)^\top \Lambda_s^{-1} (u-v)\Big),
$$

with the *pair* precision

$$
\Lambda_s = \Lambda_1 + \Lambda_2 - \Lambda_0 .
$$

The formula only involves the two atoms it pairs — nothing forces the atoms of
$f$ to share a lengthscale. Validity requires $\Lambda_s \succ 0$; atom
membership in $\mathcal{H}_0$ requires the stronger $2\Lambda_i \succ \Lambda_0$
(i.e. $\rho_i > \rho_0 / \sqrt 2$ per dimension), which implies it.

## General bilinear form

By bilinearity,

$$
\langle f, g \rangle_{\mathcal{H}_0}
= \sum_{i,j} a_i\, b_j\, c_{ij}\,
\exp\!\Big(-\tfrac12 (x_i - y_j)^\top \Lambda_{ij}^{-1} (x_i - y_j)\Big),
$$

where, per pair $(i, j)$ and with everything diagonal (elementwise on the $d$
components),

$$
\lambda_{ij} = \rho_i^2 + \rho_j'^2 - \rho_0^2,
\qquad
c_{ij} = \prod_{p=1}^{d}
\left( \frac{\rho_{i,p}^2\, \rho_{j,p}'^2}{\rho_{0,p}^2\, \lambda_{ij,p}} \right)^{1/2}.
$$

So the generalization from per-output to per-basis-point lengthscales is purely
a broadcasting change: $\lambda$ and $c$ pick up basis-point indices.

| quantity | per-output `rho: (k, d)` | per-point `rho: (k, m, d)` |
|---|---|---|
| $\lambda$ | `(k, d)` | `(k, m1, m2, d)` |
| scale $c$ | `(k,)` | `(k, m1, m2)` |
| kernel matrix | `(k, m1, m2)` | `(k, m1, m2)` |

The current implementation is the special case $\rho_i = \rho$ for all atoms of
an output: $\lambda_{ij}$ collapses to one vector per output, the scale factors
out of the double sum, and the exponential reduces to a stationary kernel whose
Gram matrix admits the usual norm-plus-gram expansion.

## Computational note

With per-point lengthscales the exponent is no longer a stationary kernel in
$x_i - y_j$ scaled by a shared $\rho$, so the $\|z_1\|^2 + \|z_2\|^2 - 2 z_1^\top z_2$
gram trick does not apply. Compute the differences explicitly:

```
diff  = x1[:, :, None, :] - x2[:, None, :, :]          # (k, m1, m2, d)
lam   = rho1[:, :, None, :]**2 + rho2[:, None, :, :]**2 - rho0**2
scale = sqrt(prod(rho1^2 * rho2^2 / (rho0^2 * lam), axis=-1))
K     = exp(-0.5 * sum(diff**2 / lam, axis=-1))
ip    = einsum("kij,ki,kj->", scale * K, a1, a2)
```

Memory is $O(k\, m_1 m_2 d)$ instead of $O(k\, m_1 m_2)$ — fine for small $m$.

## Consistency checks

- Setting all $\rho_i = \rho_j' = \rho_0$ gives $\lambda = \rho_0^2$, $c = 1$,
  recovering the plain RKHS inner product in $\mathcal{H}_0$.
- $\|f\|^2 \ge 0$ for any coefficients whenever every atom satisfies the
  membership condition, since it is a genuine Gram matrix in $\mathcal{H}_0$.
- Diagonal terms give the atom norm
  $\|k_\Lambda(\cdot, x)\|^2_{\mathcal{H}_0} =
  \prod_p \big(\rho_p^4 / (\rho_{0,p}^2 (2\rho_p^2 - \rho_{0,p}^2))\big)^{1/2}$,
  which diverges as $\rho_p \downarrow \rho_{0,p}/\sqrt2$ — the membership
  boundary.
