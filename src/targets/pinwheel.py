from typing import Callable, NamedTuple
from functools import partial
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import numpy as np

from targets import TestFunction


class Params(NamedTuple):
    """Everything the dynamics needs, as arrays, so the rollout jits once per run."""

    L1: Scalar
    L2: Scalar
    m1: Scalar
    m2: Scalar
    Lp: Scalar
    mp: Scalar
    dp: Scalar
    pivot: Float[Array, "2"]
    dr: Scalar
    kc: Scalar
    dc: Scalar
    cr: Scalar


def dynamics(
    x: Float[Array, "6"],
    refs: Float[Array, "4"],  # q1_ref, q2_ref, K1, K2 at this instant
    p: Params,
) -> Float[Array, "6"]:
    q1, q2, th, dq1, dq2, dth = x
    q1_ref, q2_ref, K1t, K2t = refs

    # ── unit vectors ──────────────────────────────────────────────
    a12 = q1 + q2

    u1 = jnp.array([jnp.cos(q1), jnp.sin(q1)])
    u12 = jnp.array([jnp.cos(a12), jnp.sin(a12)])
    up = jnp.array([jnp.cos(th), jnp.sin(th)])
    u1p = jnp.array([-jnp.sin(q1), jnp.cos(q1)])  # perp link 1
    u12p = jnp.array([-jnp.sin(a12), jnp.cos(a12)])  # perp link 2
    upp = jnp.array([-jnp.sin(th), jnp.cos(th)])  # perp pinwheel

    # ── positions ─────────────────────────────────────────────────
    P1 = p.L1 * u1
    Q = p.pivot

    # ── robot dynamics ────────────────────────────────────────────
    # the angle between the links is q2, so u1 @ u12 and u1 x u12 are just its cos/sin
    c2, s2 = jnp.cos(q2), jnp.sin(q2)

    M11 = (p.m1 / 3 + p.m2) * p.L1**2 + p.m2 * (p.L2**2 / 3 + p.L1 * p.L2 * c2)
    M12 = p.m2 * (p.L2**2 / 3 + p.L1 * p.L2 * c2 / 2)
    M22 = p.m2 * p.L2**2 / 3

    h = p.m2 * p.L1 * p.L2 * s2 / 2
    Cdq = jnp.array([-h * dq2 * dq1 - h * (dq1 + dq2) * dq2, h * dq1**2])

    tau = jnp.array(
        [
            K1t * ((q1_ref - q1) - p.dr * dq1),
            K2t * ((q2_ref - q2) - p.dr * dq2),
        ]
    )

    # ── closest point on link 2 to pinwheel ───────────────────────
    d = Q - P1
    cu = (d @ u12) / p.L2
    cup = (d @ up) / p.Lp
    cosa = u12 @ up
    sin2 = 1.0 - cosa**2

    t2 = jnp.clip(jnp.where(sin2 > 1e-10, (cu - cup * cosa) / sin2, 0.0), 0.0, 1.0)
    s = jnp.clip((t2 * p.L2 * cosa / p.Lp) - cup, 0.0, 1.0)
    t2 = jnp.clip((d @ u12 + s * p.Lp * cosa) / p.L2, 0.0, 1.0)

    ptL = P1 + t2 * p.L2 * u12
    ptP = Q + s * p.Lp * up
    gap = ptL - ptP
    dist = jnp.linalg.norm(gap)

    # ── contact force on link 2 ───────────────────────────────────
    n = gap / jnp.maximum(dist, 1e-12)
    pen = p.cr - dist
    Jv1 = p.L1 * u1p + t2 * p.L2 * u12p  # d(ptL)/d(dq1)
    Jv2 = t2 * p.L2 * u12p  # d(ptL)/d(dq2)
    v_link = Jv1 * dq1 + Jv2 * dq2
    v_pw = s * p.Lp * upp * dth
    vn = (v_link - v_pw) @ n

    F_mag = jnp.maximum(0.0, p.kc * pen - p.dc * vn)
    F = jnp.where((dist > 1e-10) & (dist < p.cr), F_mag * n, jnp.zeros(2))

    tau_arm = jnp.array([Jv1 @ F, Jv2 @ F])
    tau_pw = -(s * p.Lp * upp) @ F

    # ── equations of motion ───────────────────────────────────────
    Ip = p.mp * p.Lp**2 / 3
    # 2x2 inverse in closed form, an LU factorization per RK4 stage is not worth it
    b = tau + tau_arm - Cdq
    det = M11 * M22 - M12**2
    ddq = jnp.array([M22 * b[0] - M12 * b[1], M11 * b[1] - M12 * b[0]]) / det
    ddth = tau_pw / Ip - p.dp * dth

    return jnp.array([dq1, dq2, dth, ddq[0], ddq[1], ddth])


@jax.jit
def rollout(
    x0: Float[Array, "6"],
    refs: Float[Array, "2n+1 4"],  # sampled on the half-step grid RK4 asks for
    dt: Scalar,
    p: Params,
) -> Float[Array, "n+1 6"]:
    """Fixed-step RK4 over the whole horizon in one dispatch.

    The reference signals arrive pre-sampled, so nothing in here depends on the
    candidate function and the trace is reused across candidates and basis sizes.
    """

    def step(x, ref_triple):
        ref_a, ref_b, ref_c = ref_triple
        k1 = dynamics(x, ref_a, p)
        k2 = dynamics(x + dt / 2 * k1, ref_b, p)
        k3 = dynamics(x + dt / 2 * k2, ref_b, p)
        k4 = dynamics(x + dt * k3, ref_c, p)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x, x

    _, xs = jax.lax.scan(step, x0, (refs[0:-2:2], refs[1:-1:2], refs[2::2]))
    return jnp.concatenate([x0[None], xs])


class PinWheel:
    d: int = 1
    """
    2-link impedance-controlled arm interacting with a pinwheel.

    Parameters
    ----------
    robot_arm_lengths : array-like (2,)
        Lengths of the 2 robot arm links [m]. Default [0.5, 0.4].
    robot_arm_masses : array-like (2,)
        Masses of the 2 robot arm links [kg]. Default [2.0, 1.5].
    pinwheel_arm_length : float
        Length of the pinwheel arm [m]. Default 0.5.
    pinwheel_arm_mass : float
        Mass of the pinwheel arm [kg]. Default 5.0.
    pivot : array-like (2,)
        Pinwheel pivot position [m]. Default [0.6, 0.0].
    damping_ratio : float
        D = damping_ratio * K. Damping is proportional to stiffness for critical damping. Default 0.1.
    contact_penalty_stiffness : float
        Stiffness for penalty-based contact forces [N/m]. Default 1e6.
    d_contact : float
        Damping for contact forces [N/(m/s)]. Default 1000.
    contact_radius : float
        Effective radius for contact between arm and pinwheel [m]. Default 0.005.
    simulation_time : float
        Total simulation time [s]. Default 5.0.
    dt : float
        Integrator step [s]. Must stay well under the contact timescale
        sqrt(m / contact_penalty_stiffness). Default 1e-4.
    """

    def __init__(
        self,
        robot_arm_lengths: tuple[float, float] = (0.5, 0.4),
        robot_arm_masses: tuple[float, float] = (2.0, 1.5),
        pinwheel_arm_length: float = 0.5,
        pinwheel_arm_mass: float = 5.0,
        pinwheel_damping: float = 3.9,
        pivot: tuple[float, float] = (1.2, 0.0),
        damping_ratio: float = 0.1,
        contact_penalty_stiffness: float = 1e6,
        d_contact: float = 1000,
        contact_radius: float = 0.005,
        simulation_time: float = 3.0,
        target_angle: float = 0.0,
        dt: float = 2e-4,
    ):
        self.L1, self.L2 = robot_arm_lengths
        self.m1, self.m2 = robot_arm_masses
        self.Lp = pinwheel_arm_length
        self.mp = pinwheel_arm_mass
        self.dp = pinwheel_damping
        self.pivot = jnp.array(pivot, dtype=float)
        self.dr = damping_ratio
        self.kc = contact_penalty_stiffness
        self.dc = d_contact
        self.cr = contact_radius
        self.simulation_time = simulation_time
        self.target_angle = target_angle
        self.dt = dt
        self.n_steps = round(simulation_time / dt)

    @property
    def params(self) -> Params:
        return Params(
            L1=jnp.asarray(self.L1, float),
            L2=jnp.asarray(self.L2, float),
            m1=jnp.asarray(self.m1, float),
            m2=jnp.asarray(self.m2, float),
            Lp=jnp.asarray(self.Lp, float),
            mp=jnp.asarray(self.mp, float),
            dp=jnp.asarray(self.dp, float),
            pivot=self.pivot,
            dr=jnp.asarray(self.dr, float),
            kc=jnp.asarray(self.kc, float),
            dc=jnp.asarray(self.dc, float),
            cr=jnp.asarray(self.cr, float),
        )

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        q1 = lambda t: f(jnp.array([t / self.simulation_time]))
        _, xs = self.simulate(q1_ref=q1)
        theta_final = xs[-1, 2]
        cost = 2 * (1 - jnp.cos(theta_final - self.target_angle))
        # a diverged rollout scores the worst case rather than poisoning the surrogate
        return jnp.where(jnp.isnan(cost), 4.0, cost)

    def simulate(
        self,
        K1=lambda t: 100.0,  # stiffness for joint 1
        K2=lambda t: 20.0,  # stiffness for joint 2
        q1_ref=lambda t: 0.0,  # reference trajectory for joint 1
        q2_ref=lambda t: 0.0,  # reference trajectory for joint 2
    ) -> tuple[Float[Array, "n+1"], Float[Array, "n+1 6"]]:
        # the references are the only candidate-dependent part, so sample them
        # once on the half-step grid instead of calling back per RK4 stage
        ts_half = jnp.arange(2 * self.n_steps + 1) * (self.dt / 2)
        refs = jnp.stack(
            [
                jnp.broadcast_to(jnp.squeeze(jax.vmap(ref)(ts_half)), ts_half.shape)
                for ref in (q1_ref, q2_ref, K1, K2)
            ],
            axis=-1,
        )

        # Initial state: arm at reference, pinwheel at pi with zero velocity
        x0 = jnp.array([refs[0, 0], refs[0, 1], jnp.pi, 0.0, 0.0, 0.0])
        xs = rollout(x0, refs, jnp.asarray(self.dt, float), self.params)
        return jnp.arange(self.n_steps + 1) * self.dt, xs

    def animate(
        self, f: Callable[[Float[Array, "d"]], Scalar], filename="pinwheel.gif", fps=30
    ):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        q1_ref = lambda t: f(jnp.array([t / self.simulation_time]))
        t, xs = self.simulate(q1_ref=q1_ref)

        t = np.asarray(t)
        q1, q2, th = np.asarray(xs[:, 0]), np.asarray(xs[:, 1]), np.asarray(xs[:, 2])

        t_anim = np.arange(t[0], t[-1], 1 / fps)
        q1 = np.interp(t_anim, t, q1)
        q2 = np.interp(t_anim, t, q2)
        th = np.interp(t_anim, t, th)

        a12 = q1 + q2
        P0 = np.zeros((len(t_anim), 2))
        P1 = self.L1 * np.stack([np.cos(q1), np.sin(q1)], axis=1)
        P2 = P1 + self.L2 * np.stack([np.cos(a12), np.sin(a12)], axis=1)
        Q0 = np.array(self.pivot)
        Q1 = Q0 + self.Lp * np.stack([np.cos(th), np.sin(th)], axis=1)

        fig, ax = plt.subplots(figsize=(6, 6))
        margin = self.L1 + self.L2 + 0.2
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.85", lw=0.5)
        ax.axvline(0, color="0.85", lw=0.5)
        ax.plot(*Q0, "k+", ms=8, mew=1.5)
        # target pinwheel position
        Q_target = Q0 + self.Lp * np.array(
            [np.cos(self.target_angle), np.sin(self.target_angle)]
        )
        ax.plot(*Q_target, "o", ms=6, color="r", alpha=0.4)

        (link1,) = ax.plot([], [], "o-", lw=3, color="steelblue", ms=6)
        (link2,) = ax.plot([], [], "o-", lw=3, color="dodgerblue", ms=6)
        (pin,) = ax.plot([], [], "o-", lw=2.5, color="tomato", ms=5)
        time_tx = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, va="top")

        def update(i):
            link1.set_data([P0[i, 0], P1[i, 0]], [P0[i, 1], P1[i, 1]])
            link2.set_data([P1[i, 0], P2[i, 0]], [P1[i, 1], P2[i, 1]])
            pin.set_data([Q0[0], Q1[i, 0]], [Q0[1], Q1[i, 1]])
            time_tx.set_text(f"t = {t_anim[i]:.2f} s")
            return link1, link2, pin, time_tx

        all_x = np.concatenate([P1[:, 0], P2[:, 0], Q1[:, 0], [Q0[0]]])
        all_y = np.concatenate([P1[:, 1], P2[:, 1], Q1[:, 1], [Q0[1]]])
        pad = 0.2
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

        anim = FuncAnimation(
            fig, update, frames=len(t_anim), interval=1000 / fps, blit=True
        )
        anim.save(filename, writer="pillow", fps=fps)
        plt.close(fig)
        return filename
