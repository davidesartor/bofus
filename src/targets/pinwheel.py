from typing import Protocol, Callable
from jaxtyping import Float, Array, Scalar

import jax
import jax.numpy as jnp
import scipy as sp
import numpy as np


class TestFunction(Protocol):
    d: int

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar: ...


class PinWheel(TestFunction):
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

    def __call__(self, f: Callable[[Float[Array, "d"]], Scalar]) -> Scalar:
        q1 = lambda t: f(jnp.array([t / self.simulation_time]))
        try:
            sol, theta_final, omega_final = self.simulate(q1_ref=q1)
            return 2 * (1 - jnp.cos(theta_final - self.target_angle))
        except Exception as e:
            print(f"Simulation failed: {e}")
            return jnp.array(4.0)  # worst case cost if simulation fails

    def simulate(
        self,
        K1=lambda t: 100.0,  # stiffness for joint 1
        K2=lambda t: 20.0,  # stiffness for joint 2
        q1_ref=lambda t: 0.0,  # reference trajectory for joint 1
        q2_ref=lambda t: 0.0,  # reference trajectory for joint 2
    ):
        def dynamics(t, x):
            q1, q2, th, dq1, dq2, dth = x

            # ── unit vectors ──────────────────────────────────────────────
            a12 = q1 + q2

            u1 = jnp.array([jnp.cos(q1), jnp.sin(q1)])
            u12 = jnp.array([jnp.cos(a12), jnp.sin(a12)])
            up = jnp.array([jnp.cos(th), jnp.sin(th)])
            u1p = jnp.array([-jnp.sin(q1), jnp.cos(q1)])  # perp link 1
            u12p = jnp.array([-jnp.sin(a12), jnp.cos(a12)])  # perp link 2
            upp = jnp.array([-jnp.sin(th), jnp.cos(th)])  # perp pinwheel

            # ── positions ─────────────────────────────────────────────────
            P1 = self.L1 * u1
            Q = self.pivot

            # ── robot dynamics ────────────────────────────────────────────
            c2 = u1 @ u12
            s2 = jnp.cross(u1, u12)

            M11 = (self.m1 / 3 + self.m2) * self.L1**2 + self.m2 * (
                self.L2**2 / 3 + self.L1 * self.L2 * c2
            )
            M12 = self.m2 * (self.L2**2 / 3 + self.L1 * self.L2 * c2 / 2)
            M22 = self.m2 * self.L2**2 / 3
            M = jnp.array([[M11, M12], [M12, M22]])

            h = self.m2 * self.L1 * self.L2 * s2 / 2
            C = jnp.array([[-h * dq2, -h * (dq1 + dq2)], [h * dq1, 0.0]])

            K1t, K2t = K1(t), K2(t)
            tau = jnp.array(
                [
                    K1t * ((q1_ref(t) - q1) - self.dr * dq1),
                    K2t * ((q2_ref(t) - q2) - self.dr * dq2),
                ]
            )

            # ── closest point on link 2 to pinwheel ───────────────────────
            d = Q - P1
            cu = (d @ u12) / self.L2
            cup = (d @ up) / self.Lp
            cosa = u12 @ up
            sin2 = 1.0 - cosa**2

            t2 = jnp.clip(
                jnp.where(sin2 > 1e-10, (cu - cup * cosa) / sin2, 0.0), 0.0, 1.0
            )
            s = jnp.clip((t2 * self.L2 * cosa / self.Lp) - cup, 0.0, 1.0)
            t2 = jnp.clip((d @ u12 + s * self.Lp * cosa) / self.L2, 0.0, 1.0)

            ptL = P1 + t2 * self.L2 * u12
            ptP = Q + s * self.Lp * up
            gap = ptL - ptP
            dist = jnp.linalg.norm(gap)

            # ── contact force on link 2 ───────────────────────────────────
            n = gap / jnp.maximum(dist, 1e-12)
            pen = self.cr - dist
            Jv1 = self.L1 * u1p + t2 * self.L2 * u12p  # d(ptL)/d(dq1)
            Jv2 = t2 * self.L2 * u12p  # d(ptL)/d(dq2)
            v_link = Jv1 * dq1 + Jv2 * dq2
            v_pw = s * self.Lp * upp * dth
            vn = (v_link - v_pw) @ n

            F_mag = jnp.maximum(0.0, self.kc * pen - self.dc * vn)
            F = jnp.where((dist > 1e-10) & (dist < self.cr), F_mag * n, jnp.zeros(2))

            tau_arm = jnp.array([Jv1 @ F, Jv2 @ F])
            tau_pw = -(s * self.Lp * upp) @ F

            # ── equations of motion ───────────────────────────────────────
            Ip = self.mp * self.Lp**2 / 3
            ddq = jnp.linalg.solve(M, tau + tau_arm - C @ jnp.array([dq1, dq2]))
            ddth = tau_pw / Ip - self.dp * dth

            return jnp.array([dq1, dq2, dth, ddq[0], ddq[1], ddth])

        # Initial state: arm at reference, pinwheel at pi with zero velocity
        sol = sp.integrate.solve_ivp(
            fun=jax.jit(dynamics),
            t_span=[0, self.simulation_time],
            y0=[q1_ref(0), q2_ref(0), jnp.pi, 0.0, 0.0, 0.0],
            method="Radau",
            rtol=1e-4,
            atol=1e-4,
            max_step=5e-3,
        )
        _, _, theta_final, _, _, omega_final = sol.y[:, -1]
        return sol, theta_final, omega_final

    def animate(
        self, f: Callable[[Float[Array, "d"]], Scalar], filename="pinwheel.gif", fps=30
    ):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        q1_ref = lambda t: f(jnp.array([t / self.simulation_time]))
        sol, _, _ = self.simulate(q1_ref=q1_ref)

        t = sol.t
        q1, q2, th = sol.y[0], sol.y[1], sol.y[2]

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
