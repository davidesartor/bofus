"""4-DOF arm striking a target with a 25-node whip (Nah et al., iScience 2023)."""

from typing import Callable, NamedTuple
from jaxtyping import Array, Float, Int, Scalar

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from einops import einsum


class WhipParams(NamedTuple):
    """Hatze upper-limb segments and the experimentally fitted bullwhip of the paper."""

    arm_length: Float[Array, "2"] = jnp.array([0.294, 0.291])
    arm_mass: Float[Array, "2"] = jnp.array([1.595, 0.869])
    arm_com: Float[Array, "2"] = jnp.array([0.129, 0.112])
    arm_inertia: Float[Array, "2 3"] = jnp.array(
        [[0.011917, 0.011937, 0.001325], [0.004765, 0.004855, 0.000472]]
    )
    node_mass: float = 0.012
    segment_length: float = 0.072
    stiffness: float = 0.242
    damping: float = 0.092
    gravity: float = 9.81


class Chain(NamedTuple):
    """Static structure of the serial chain: hinge axes, the link after each hinge, the bodies."""

    axes: Float[Array, "n 3"]
    offsets: Float[Array, "n 3"]
    masses: Float[Array, "m"]
    com_offsets: Float[Array, "m 3"]
    attached_to: Int[Array, "m"]


def build_chain(n_nodes: int, p: WhipParams) -> Chain:
    """Shoulder (flexion -y, adduction -x, rotation +z), elbow (-y), then (+x, -y) per whip node.

    Bodies are the two arm links, the massless hand (for gravity compensation)
    and the whip point masses, each attached after the hinge index in `attached_to`.
    """
    n = 4 + 2 * n_nodes
    axes = np.array(
        [[0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, -1, 0]]
        + [[1, 0, 0], [0, -1, 0]] * n_nodes
    )
    lengths = np.zeros(n)
    lengths[[2, 3]] = p.arm_length
    lengths[5::2] = p.segment_length
    offsets = np.zeros((n, 3))
    offsets[:, 2] = -lengths

    masses = np.concatenate([p.arm_mass, [0.0], np.full(n_nodes, p.node_mass)])
    com_offsets = np.zeros((3 + n_nodes, 3))
    com_offsets[:2, 2] = -np.asarray(p.arm_com)
    com_offsets[2, 2] = -p.arm_length[1]
    com_offsets[3:, 2] = -p.segment_length
    attached_to = np.concatenate([[2, 3, 3], 5 + 2 * np.arange(n_nodes)])
    return Chain(*map(jnp.asarray, (axes, offsets, masses, com_offsets, attached_to)))


def rodrigues(axis: Float[Array, "3"], angle: Scalar) -> Float[Array, "3 3"]:
    K = jnp.cross(jnp.eye(3), axis)
    return jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * K @ K


def kinematics(q: Float[Array, "n"], chain: Chain):
    """World hinge axes and origins, orientation after each hinge, body positions."""
    R = jax.lax.associative_scan(jnp.matmul, jax.vmap(rodrigues)(chain.axes, q))
    R_before = jnp.concatenate([jnp.eye(3)[None], R[:-1]])
    axes = einsum(R_before, chain.axes, "j a b, j b -> j a")
    steps = einsum(R, chain.offsets, "j a b, j b -> j a")
    ends = jnp.cumsum(steps, axis=0)
    origins = ends - steps
    R_body = R[chain.attached_to]
    positions = origins[chain.attached_to] + einsum(
        R_body, chain.com_offsets, "m a b, m b -> m a"
    )
    return axes, origins, R_body, positions


def jacobians(q: Float[Array, "n"], chain: Chain):
    """Linear Jacobian of every body, angular Jacobian and orientation of the two arm links."""
    axes, origins, R_body, positions = kinematics(q, chain)
    upstream = jnp.arange(q.shape[0])[None, :] <= chain.attached_to[:, None]
    lever = positions[:, None] - origins[None]
    Jv = upstream[:, :, None] * jnp.cross(axes[None], lever)
    Jw = upstream[:2, :, None] * axes[None]
    return Jv, Jw, R_body[:2]


def node_positions(q: Float[Array, "n"], chain: Chain) -> Float[Array, "n/2-2 3"]:
    return kinematics(q, chain)[3][3:]


def joint_positions(q: Float[Array, "n"], chain: Chain) -> Float[Array, "n/2+1 3"]:
    """Shoulder, elbow, hand and the whip nodes, for plotting."""
    _, origins, _, positions = kinematics(q, chain)
    return jnp.concatenate([origins[jnp.array([0, 3])], positions[2:]])


def dynamics_terms(
    q: Float[Array, "n"], dq: Float[Array, "n"], chain: Chain, p: WhipParams
) -> tuple[
    Float[Array, "n n"], Float[Array, "n"], Float[Array, "n"], Float[Array, "4"]
]:
    """Mass matrix, velocity-product forces, gravity forces and the arm's gravity compensation."""
    (Jv, Jw, R), (dJv, dJw, _) = jax.jvp(lambda q: jacobians(q, chain), (q,), (dq,))
    m = chain.masses
    inertia = einsum(R, p.arm_inertia, R, "i a c, i c, i b c -> i a b")

    M = einsum(Jv, m, Jv, "m j a, m, m k a -> j k")
    M = M + einsum(Jw, inertia, Jw, "i j a, i a b, i k b -> j k")

    # forces from the acceleration each body has at zero joint acceleration
    omega = einsum(Jw, dq, "i j a, j -> i a")
    alpha = einsum(dJw, dq, "i j a, j -> i a")
    gyro = einsum(inertia, alpha, "i a b, i b -> i a")
    gyro = gyro + jnp.cross(omega, einsum(inertia, omega, "i a b, i b -> i a"))
    bias = einsum(Jv, m, einsum(dJv, dq, "m j a, j -> m a"), "m j a, m, m a -> j")
    bias = bias + einsum(Jw, gyro, "i j a, i a -> j")

    gravity = -p.gravity * einsum(Jv[:, :, 2], m, "m j, m -> j")
    # the original compensates the arm links plus the whole whip mass lumped at the hand
    lumped = m.at[2].set(m[3:].sum()).at[3:].set(0.0)
    compensation = p.gravity * einsum(Jv[:, :4, 2], lumped, "m j, m -> j")
    return M, bias, gravity, compensation


def hanging_whip_angles(q_arm: Float[Array, "4"], chain: Chain) -> Float[Array, "2"]:
    """First universal joint angles that point the whip straight down from the hand."""
    q = jnp.zeros(chain.axes.shape[0]).at[:4].set(q_arm)
    R_forearm = kinematics(q, chain)[2][1]
    vx, vy, vz = R_forearm.T @ jnp.array([0.0, 0.0, -1.0])
    return jnp.array([jnp.arctan2(vy, -vz), jnp.arcsin(vx)])


def target_position(index: int, reach: float, offset: float = 0.4) -> Float[Array, "3"]:
    """The paper's six targets in (radius, azimuth, elevation) around the shoulder."""
    radius = reach - offset if index <= 3 else reach
    azimuth, elevation = [(0.0, 0.0), (45.0, 0.0), (45.0, 45.0)][(index - 1) % 3]
    azimuth, elevation = np.deg2rad(azimuth), np.deg2rad(elevation)
    direction = [
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ]
    return radius * jnp.array(direction)


PAPER_MIN_JERK = {
    1: ((-1.3614, 0.0, -0.3491, 1.4137), (1.7279, 0.0, 0.0, 1.4137), 0.9500),
    2: ((-0.9425, 0.0, -1.0472, 1.4137), (1.7279, -1.0472, 0.0, 0.4712), 0.9500),
    3: ((-0.9425, 1.0472, 0.0, 1.4137), (1.7279, -1.0472, 0.0, 1.4137), 0.5833),
    4: ((-1.5475, 0.0, -0.3491, 1.4137), (1.7279, 0.0, 0.0, 0.3665), 0.9500),
    5: ((-1.0821, 1.0472, 1.0472, 0.8203), (1.7279, -1.0472, 1.3963, 0.1571), 0.9500),
    6: ((-0.9425, 1.0601, 0.3491, 0.9948), (1.7395, -0.9696, -0.2715, 0.9483), 0.5245),
}
"""Table 1 of the paper: initial posture, final posture and duration of the best min-jerk move."""


def min_jerk(
    ts: Float[Array, "t"], q_start: Float[Array, "4"], q_end: Float[Array, "4"], duration: float
) -> tuple[Float[Array, "t 4"], Float[Array, "t 4"]]:
    """Minimum-jerk move from q_start to q_end starting at t=0, held still before and after."""
    tau = jnp.clip(ts / duration, 0.0, 1.0)[:, None]
    shape = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    rate = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    return q_start + (q_end - q_start) * shape, (q_end - q_start) * rate


class Whip:
    """Hit a target with a whip held by a joint-impedance-controlled 4-DOF arm.

    The four outputs of f are the zero-torque joint trajectories over normalized
    time (Equation 1 of the paper). The cost is the smallest distance the last
    `n_tip_nodes` whip nodes ever get to the target, minus the contact radius.
    """

    d: int = 1  # normalized time
    k: int = 4  # shoulder flexion, adduction, rotation and elbow

    def __init__(
        self,
        params: WhipParams = WhipParams(),
        n_nodes: int = 25,
        target: int | tuple[float, float, float] = 1,
        hit_radius: float = 0.05,
        n_tip_nodes: int = 5,
        stiffness_matrix: tuple[tuple[float, ...], ...] = (
            (17.4, 6.85, -7.75, 8.40),
            (6.85, 33.0, 3.70, 0.00),
            (-7.75, 3.70, 27.7, 0.00),
            (8.40, 0.00, 0.00, 23.2),
        ),
        damping_ratio: float = 0.05,
        reference_center: tuple[float, ...] = (0.25 * np.pi, 0.0, 0.0, 0.45 * np.pi),
        reference_amplitude: tuple[float, ...] = (
            0.75 * np.pi,
            0.5 * np.pi,
            0.5 * np.pi,
            0.45 * np.pi,
        ),
        horizon: float = 2.5,
        dt: float = 5e-4,
    ):
        self.params = params
        self.n_nodes = n_nodes
        self.chain = build_chain(n_nodes, params)
        self.reach = float(jnp.sum(params.arm_length)) + n_nodes * params.segment_length
        if isinstance(target, int):
            target = target_position(target, self.reach)
        self.target = jnp.asarray(target, float)
        self.hit_radius = hit_radius
        self.n_tip_nodes = n_tip_nodes
        self.Kq = jnp.asarray(stiffness_matrix, float)
        self.Bq = damping_ratio * self.Kq
        self.reference_center = jnp.asarray(reference_center, float)
        self.reference_amplitude = jnp.asarray(reference_amplitude, float)
        self.horizon = horizon
        self.dt = dt

    @property
    def n_steps(self) -> int:
        return round(self.horizon / self.dt)

    def __call__(self, f: Callable[[Float[Array, "1"]], Float[Array, "4"]]) -> Scalar:
        _, q0, dq0 = self.reference(f)
        _, qs, _ = self.rollout(q0, dq0)
        cost = self.min_distance(qs)
        # a diverged rollout scores the worst case rather than poisoning the surrogate
        return jnp.where(jnp.isnan(cost), 2 * self.reach, cost)

    def reference(
        self, f: Callable[[Float[Array, "1"]], Float[Array, "4"]]
    ) -> tuple[Float[Array, "t"], Float[Array, "t 4"], Float[Array, "t 4"]]:
        """Zero-torque joint trajectory and its rate at every step."""

        def q0(t: Scalar) -> Float[Array, "4"]:
            f_t = jnp.reshape(f(jnp.array([t / self.horizon])), (4,))
            return self.reference_center + self.reference_amplitude * f_t

        ts = jnp.arange(self.n_steps) * self.dt
        q0s, dq0s = jax.vmap(lambda t: (q0(t), jax.jacfwd(q0)(t)))(ts)
        return ts, q0s, dq0s

    def paper_reference(
        self, index: int
    ) -> tuple[Float[Array, "t"], Float[Array, "t 4"], Float[Array, "t 4"]]:
        """The paper's best min-jerk move for target `index` (Table 1), sampled at every step."""
        q_start, q_end, duration = (jnp.asarray(v, float) for v in PAPER_MIN_JERK[index])
        ts = jnp.arange(self.n_steps) * self.dt
        q0s, dq0s = min_jerk(ts, q_start, q_end, float(duration))
        return ts, q0s, dq0s

    @eqx.filter_jit
    def rollout(
        self, q0: Float[Array, "t 4"], dq0: Float[Array, "t 4"]
    ) -> tuple[Float[Array, "t+1"], Float[Array, "t+1 n"], Float[Array, "t+1 n"]]:
        """Semi-implicit Euler with implicit joint damping (as MuJoCo), from rest, whip hanging."""
        p, chain = self.params, self.chain
        n = 4 + 2 * self.n_nodes
        whip_stiffness = jnp.full(2 * self.n_nodes, p.stiffness).at[:2].set(0.0)
        whip_damping = jnp.full(2 * self.n_nodes, p.damping).at[:2].set(0.0)
        D = jnp.zeros((n, n)).at[:4, :4].set(self.Bq) + jnp.diag(
            jnp.pad(whip_damping, (4, 0))
        )

        def step(state, ref):
            q, dq = state
            q0_t, dq0_t = ref
            M, bias, gravity, compensation = dynamics_terms(q, dq, chain, p)
            tau_arm = self.Kq @ (q0_t - q[:4]) + self.Bq @ dq0_t + compensation
            tau = jnp.concatenate([tau_arm, -whip_stiffness * q[4:]]) - D @ dq
            ddq = jnp.linalg.solve(M + self.dt * D, tau + gravity - bias)
            dq = dq + self.dt * ddq
            q = q + self.dt * dq
            return (q, dq), (q, dq)

        q_init = jnp.zeros(n).at[:4].set(q0[0])
        q_init = q_init.at[4:6].set(hanging_whip_angles(q0[0], chain))
        _, (qs, dqs) = jax.lax.scan(step, (q_init, jnp.zeros(n)), (q0, dq0))
        ts = jnp.arange(self.n_steps + 1) * self.dt
        qs = jnp.concatenate([q_init[None], qs])
        dqs = jnp.concatenate([jnp.zeros((1, n)), dqs])
        return ts, qs, dqs

    @eqx.filter_jit
    def min_distance(self, qs: Float[Array, "t n"]) -> Scalar:
        """Closest the tip nodes get to the target over the rollout, zero once touching."""
        nodes = jax.vmap(node_positions, in_axes=(0, None))(qs, self.chain)
        dist = jnp.linalg.norm(nodes[:, -self.n_tip_nodes :] - self.target, axis=-1)
        return jnp.clip(dist.min() - self.hit_radius, min=0.0)

    def animate(
        self,
        f: Callable[[Float[Array, "1"]], Float[Array, "4"]],
        filename="whip.gif",
        fps=30,
        view=(20.0, -60.0),
        figsize=(7.0, 6.0),
    ):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        _, q0, dq0 = self.reference(f)
        t, qs, _ = self.rollout(q0, dq0)
        pts = np.asarray(jax.vmap(joint_positions, in_axes=(0, None))(qs, self.chain))
        t = np.asarray(t)

        t_anim = np.arange(t[0], t[-1], 1 / fps)
        idx = np.searchsorted(t, t_anim)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
        r = self.reach
        ax.set_xlim(-r, r), ax.set_ylim(-r, r), ax.set_zlim(-r, r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*view)
        ax.scatter(*np.asarray(self.target), s=40, color="r", alpha=0.6)
        (arm,) = ax.plot([], [], [], "o-", lw=3, color="steelblue", ms=5)
        (whip,) = ax.plot([], [], [], "-", lw=1.5, color="magenta")
        time_tx = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=9)

        def update(i):
            j = idx[i]
            arm.set_data(pts[j, :3, 0], pts[j, :3, 1])
            arm.set_3d_properties(pts[j, :3, 2])
            whip.set_data(pts[j, 2:, 0], pts[j, 2:, 1])
            whip.set_3d_properties(pts[j, 2:, 2])
            time_tx.set_text(f"t = {t_anim[i]:.2f} s")
            return arm, whip, time_tx

        anim = FuncAnimation(fig, update, frames=len(t_anim), interval=1000 / fps)
        anim.save(filename, writer="pillow", fps=fps)
        plt.close(fig)
        return filename
