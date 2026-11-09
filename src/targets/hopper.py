"""Planar single-leg hopping robot with rigid ground contact."""

from typing import Callable, NamedTuple
from jaxtyping import Array, Bool, Float, Scalar

import jax
import jax.numpy as jnp
import equinox as eqx


class RobotParams(NamedTuple):
    mass: Float[Array, "4"] = jnp.array(
        [3.0, 1.5, 1.0, 2.0]
    )  # body, thigh, shank, foot
    inertia: Float[Array, "4"] = jnp.array([0.01, 0.005, 0.005, 0.009])
    length: Float[Array, "4"] = jnp.array([0.3, 0.6, 0.6, 0.5])  # l0 .. l3
    com_offset: Float[Array, "3"] = jnp.array([0.3, 0.2, 0.4])  # c1 .. c3
    gravity: float = 9.81


def link_directions(q: Float[Array, "6"]) -> Float[Array, "4 2"]:
    """Unit vectors along body, thigh, shank and foot."""
    _, _, thb, th1, th2, th3 = q
    angles = jnp.cumsum(jnp.array([thb, th1, th2]))
    along = jnp.stack([jnp.sin(angles), -jnp.cos(angles)], axis=-1)
    foot = angles[-1] + th3
    return jnp.concatenate([along, jnp.array([[jnp.cos(foot), jnp.sin(foot)]])])


def keypoints(q: Float[Array, "6"], p: RobotParams) -> Float[Array, "6 2"]:
    """Base, hip, knee, ankle, toe and heel positions."""
    e = link_directions(q)
    base = q[:2]
    hip = base + p.length[0] * e[0]
    knee = hip + p.length[1] * e[1]
    ankle = knee + p.length[2] * e[2]
    toe = ankle + 0.5 * p.length[3] * e[3]
    heel = ankle - 0.5 * p.length[3] * e[3]
    return jnp.stack([base, hip, knee, ankle, toe, heel])


def com_positions(q: Float[Array, "6"], p: RobotParams) -> Float[Array, "4 2"]:
    e = link_directions(q)
    base, hip, knee, ankle, _, _ = keypoints(q, p)
    return jnp.stack(
        [
            base,
            hip + p.com_offset[0] * e[1],
            knee + p.com_offset[1] * e[2],
            ankle,
        ]
    )


def link_angles(q: Float[Array, "6"]) -> Float[Array, "4"]:
    return q[2] + jnp.concatenate([jnp.zeros(1), jnp.cumsum(q[3:])])


def lagrangian(q: Float[Array, "6"], dq: Float[Array, "6"], p: RobotParams) -> Scalar:
    v = jax.jacobian(com_positions)(q, p) @ dq
    w = jax.jacobian(link_angles)(q) @ dq
    kinetic = 0.5 * jnp.sum(p.mass * jnp.sum(v**2, axis=-1)) + 0.5 * jnp.sum(
        p.inertia * w**2
    )
    potential = p.gravity * jnp.sum(p.mass * com_positions(q, p)[:, 1])
    return kinetic - potential


def mass_matrix(q: Float[Array, "6"], p: RobotParams) -> Float[Array, "6 6"]:
    return jax.hessian(lagrangian, argnums=1)(q, jnp.zeros_like(q), p)


def forward_dynamics(
    q: Float[Array, "6"], dq: Float[Array, "6"], tau: Float[Array, "3"], p: RobotParams
) -> Float[Array, "6"]:
    """Joint accelerations from d/dt(dL/ddq) - dL/dq = Q."""
    dL_dq = jax.grad(lagrangian, argnums=0)(q, dq, p)
    dp_dq = jax.jacobian(jax.grad(lagrangian, argnums=1), argnums=0)(q, dq, p)
    generalized_force = jnp.concatenate([jnp.zeros(3), tau])
    return jnp.linalg.solve(mass_matrix(q, p), generalized_force + dL_dq - dp_dq @ dq)


def contact_jacobians(q: Float[Array, "6"], p: RobotParams) -> Float[Array, "2 2 6"]:
    """Jacobians of the toe and heel positions."""
    return jax.jacobian(keypoints)(q, p)[4:]


def resolve_contacts(
    q: Float[Array, "6"],
    dq: Float[Array, "6"],
    p: RobotParams,
    restitution: float,
    friction: float,
    iterations: int = 20,
) -> Float[Array, "6"]:
    """Project the velocity onto the contact constraints (Gauss-Seidel impulses)."""
    Minv = jnp.linalg.inv(mass_matrix(q, p))
    contacts = keypoints(q, p)[4:]
    J = contact_jacobians(q, p)

    # a penetrating point bounces back at -restitution times its approach speed
    active = contacts[:, 1] < 0.0
    v = jnp.einsum("cij,j->ci", J, dq)
    target_v = jnp.stack(
        [jnp.zeros(2), jnp.where(v[:, 1] < 0, -restitution, 1) * v[:, 1]], -1
    )
    inverse_inertia = jnp.einsum("cij,jk,cik->ci", J, Minv, J)

    def apply_impulse(dq, total, contact, axis, bound):
        """One Gauss-Seidel sweep of a single contact direction."""
        Jc = J[contact, axis]
        increment = (target_v[contact, axis] - Jc @ dq) / inverse_inertia[contact, axis]
        increment = jnp.where(active[contact], increment, 0.0)
        clamped = jnp.clip(total[contact, axis] + increment, *bound(total))
        increment = clamped - total[contact, axis]
        return dq + Minv @ Jc * increment, total.at[contact, axis].add(increment)

    def sweep(_, state):
        dq, total = state
        # normal impulses are compressive, tangential ones live inside the friction cone
        for contact in range(2):
            dq, total = apply_impulse(
                dq, total, contact, 1, lambda total: (0.0, jnp.inf)
            )
            limit = lambda total: friction * total[contact, 1]
            dq, total = apply_impulse(
                dq, total, contact, 0, lambda total: (-limit(total), limit(total))
            )
        return dq, total

    dq, _ = jax.lax.fori_loop(0, iterations, sweep, (dq, jnp.zeros((2, 2))))
    return dq


class HoppingRobot:
    """Jump as high as possible with the hip and knee, holding the ankle in place.

    The two outputs of f are desired hip and knee offsets over normalized time,
    tracked by a joint-space PD controller; the ankle is held at its initial angle.
    """

    d: int = 1  # normalized time
    k: int = 2  # hip and knee

    def __init__(
        self,
        initial_state: Float[Array, "12"] = jnp.array(
            [0.0, 1.4, 0.0, 0.8, -1.6, 0.8] + [0.0] * 6
        ),
        params: RobotParams = RobotParams(),
        horizon: float = 2.0,
        dt: float = 1e-3,
        amplitude: float = 0.8,
        torque_limit: float = 100.0,
        kp: float = 250.0,
        kd: float = 10.0,
        restitution: float = 0.0,
        friction: float = 0.7,
    ):
        self.initial_state = initial_state
        self.params = params
        self.horizon = horizon
        self.dt = dt
        self.amplitude = amplitude
        self.torque_limit = torque_limit
        self.kp = kp
        self.kd = kd
        self.restitution = restitution
        self.friction = friction

    @property
    def n_steps(self) -> int:
        return int(self.horizon / self.dt)

    def __call__(self, f: Callable[[Float[Array, "1"]], Float[Array, "2"]]) -> Scalar:
        qs, _, grounded = self.rollout(f)

        # only count the apex reached after the robot has landed at least once
        height = jnp.where(grounded.cumsum() > 0, qs[:, 1], -jnp.inf)

        # a diverged rollout scores worse than any height the robot can reach
        return jnp.nan_to_num(-height.max(), nan=0.0, posinf=0.0, neginf=0.0)

    @eqx.filter_jit
    def rollout(
        self, f: Callable[[Float[Array, "1"]], Float[Array, "2"]]
    ) -> tuple[Float[Array, "t 6"], Float[Array, "t 6"], Bool[Array, "t"]]:
        p = self.params
        q0, dq0 = self.initial_state[:6], self.initial_state[6:]

        # the desired joint trajectory does not depend on the state, so precompute it
        ts = jnp.arange(self.n_steps) / self.n_steps
        q_des = q0[3:] + self.amplitude * jnp.pad(
            jax.vmap(f)(ts[:, None]), ((0, 0), (0, 1))
        )

        def step(state, q_des):
            q, dq = state
            tau = self.kp * (q_des - q[3:]) - self.kd * dq[3:]
            tau = jnp.clip(tau, -self.torque_limit, self.torque_limit)

            # semi-implicit euler, with the contact impulses applied to the new velocity
            dq = dq + self.dt * forward_dynamics(q, dq, tau, p)
            dq = resolve_contacts(q, dq, p, self.restitution, self.friction)
            q = q + self.dt * dq
            grounded = jnp.any(keypoints(q, p)[4:, 1] < 0.0)
            return (q, dq), (q, dq, grounded)

        _, trajectory = jax.lax.scan(step, (q0, dq0), q_des)
        return trajectory
