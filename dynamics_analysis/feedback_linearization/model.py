"""
26-DOF cooperative pinch-lift-move model (two Go2 floating bases + D1 4-DOF arms
+ rigid payload), implemented as the control-affine constrained Lagrangian system
described in system_model_notes.txt.

Generalized coordinates  q in R^26, ordered exactly as in the report
    q = ( q1_b ,  th1_a ,  q2_b ,  th2_a ,  p )
indices:
    0:3    robot 1 base position   (x1, y1, z1)
    3:6    robot 1 base ZYX-Euler  (phi1, theta1, psi1)   (roll, pitch, yaw)
    6:10   robot 1 arm joints      (th1_1 .. th1_4)
    10:13  robot 2 base position
    13:16  robot 2 base ZYX-Euler
    16:20  robot 2 arm joints
    20:23  payload position        (px, py, pz)
    23:26  payload ZYX-Euler       (alpha, beta, gamma)

The model provides
    D(q)            block-diagonal generalized inertia (26x26, SPD)
    G(q)            gravity generalized force            (26,)
    coriolis(q,qd)  C(q,qd) qd                           (26,)   [Christoffel, finite diff]
    B               actuation matrix (26x8) selecting the 8 arm torques
    constraints(q)  holonomic contact constraints kappa  (12,)   (both EEs pinned to payload)
    output(q,t)     payload-pose tracking error y        (6,)
plus the various Jacobians used by the feedback-linearizing controller.

Everything is plain numpy so the study runs without Isaac Lab.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Indexing helpers
# ----------------------------------------------------------------------------
NQ = 26
# slices for the two robots and the payload
R1 = dict(base_pos=slice(0, 3),  base_eul=slice(3, 6),  arm=slice(6, 10),  all=slice(0, 10))
R2 = dict(base_pos=slice(10, 13), base_eul=slice(13, 16), arm=slice(16, 20), all=slice(10, 20))
PAY = dict(pos=slice(20, 23), eul=slice(23, 26), all=slice(20, 26))
ROBOTS = (R1, R2)

g = 9.81

# ----------------------------------------------------------------------------
# Physical parameters (Go2 base + small D1-scale 4R arm + box payload).
# Values are representative; the structural conclusions (relative degree,
# output convergence, zero-dynamics stability) do not hinge on exact numbers.
# ----------------------------------------------------------------------------
M_BASE = 13.0                                   # Go2 trunk mass [kg]
I_BASE = np.diag([0.12, 0.40, 0.42])            # trunk inertia in body frame [kg m^2]

ARM_MOUNT = np.array([0.18, 0.0, 0.06])         # arm mount offset in base frame [m]
L1, L2, L3, L4 = 0.10, 0.16, 0.16, 0.10         # D1-scale link lengths [m]
M_LINK = 0.45                                   # per arm-link mass [kg]
I_LINK = np.diag([4e-3, 4e-3, 4e-3])            # per arm-link inertia [kg m^2]
M_EE = 0.30                                      # contact-pad mass [kg]
I_EE = np.diag([2e-3, 2e-3, 2e-3])

M_PAY = 3.0                                       # payload mass [kg]
BOX = np.array([0.30, 0.40, 0.25])                # payload full extents (x,y,z) [m]
# box inertia about its COM
I_PAY = (M_PAY / 12.0) * np.diag([
    BOX[1] ** 2 + BOX[2] ** 2,
    BOX[0] ** 2 + BOX[2] ** 2,
    BOX[0] ** 2 + BOX[1] ** 2,
])

# Contact frames on the payload (in payload body frame). Robot 1 pinches the -x
# face, robot 2 the +x face.  R_off orients the pad so its +z (pushing) axis
# points inward toward the payload centre.
CONTACT_OFFSET = (np.array([-BOX[0] / 2, 0.0, 0.0]),
                  np.array([+BOX[0] / 2, 0.0, 0.0]))


# ----------------------------------------------------------------------------
# SO(3) utilities
# ----------------------------------------------------------------------------
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_zyx_to_R(eul):
    """R = Rz(psi) Ry(theta) Rx(phi) for eul = (phi, theta, psi)."""
    phi, theta, psi = eul
    return Rz(psi) @ Ry(theta) @ Rx(phi)


def W_euler(eul):
    """Map ZYX Euler rates (phi_dot,theta_dot,psi_dot) -> body angular velocity.
    Matches W_r in the report.  Singular at theta = +-pi/2 (gimbal lock)."""
    phi, theta, _ = eul
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, -sth],
                     [0.0, cph, cth * sph],
                     [0.0, -sph, cth * cph]])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def log_SO3(R):
    """Matrix log of a rotation -> rotation vector in R^3 (axis*angle)."""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-9:
        # first-order: vee of skew-symmetric part
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


# ----------------------------------------------------------------------------
# Per-robot kinematics: build all body frames / COMs / joint axes from q_r.
# ----------------------------------------------------------------------------
def robot_kinematics(qr):
    """Given the 10 coordinates of one robot, return a dict of body poses,
    COM positions, joint axes & origins (all in world frame).

    Arm structure (a yaw + three parallel pitches, D1-like):
        joint1 : axis z  at mount,        link L1 along +z
        joint2 : axis y,                  link L2 along +x
        joint3 : axis y,                  link L3 along +x
        joint4 : axis y,                  link L4 along +x  -> end-effector pad
    """
    b = qr[0:3]
    eul = qr[3:6]
    th = qr[6:10]
    Rb = euler_zyx_to_R(eul)

    # mount frame
    o_m = b + Rb @ ARM_MOUNT
    R_m = Rb

    # joint 1 (z)
    a1 = R_m @ np.array([0, 0, 1.0])
    o_j1 = o_m
    Rj1 = R_m @ Rz(th[0])
    o1 = o_j1 + Rj1 @ np.array([0, 0, L1])
    # joint 2 (y)
    a2 = Rj1 @ np.array([0, 1.0, 0])
    o_j2 = o1
    Rj2 = Rj1 @ Ry(th[1])
    o2 = o_j2 + Rj2 @ np.array([L2, 0, 0])
    # joint 3 (y)
    a3 = Rj2 @ np.array([0, 1.0, 0])
    o_j3 = o2
    Rj3 = Rj2 @ Ry(th[2])
    o3 = o_j3 + Rj3 @ np.array([L3, 0, 0])
    # joint 4 (y)
    a4 = Rj3 @ np.array([0, 1.0, 0])
    o_j4 = o3
    Rj4 = Rj3 @ Ry(th[3])
    o_ee = o_j4 + Rj4 @ np.array([L4, 0, 0])
    R_ee = Rj4

    # bodies: name -> (mass, inertia_local, R_world, com_world,
    #                  highest arm-joint index that moves this body [0..4])
    bodies = []
    bodies.append(("base", M_BASE, I_BASE, Rb, b.copy(), 0))
    bodies.append(("l1", M_LINK, I_LINK, Rj1, 0.5 * (o_j1 + o1), 1))
    bodies.append(("l2", M_LINK, I_LINK, Rj2, 0.5 * (o_j2 + o2), 2))
    bodies.append(("l3", M_LINK, I_LINK, Rj3, 0.5 * (o_j3 + o3), 3))
    bodies.append(("l4", M_LINK, I_LINK, Rj4, 0.5 * (o_j4 + o_ee), 4))
    bodies.append(("ee", M_EE, I_EE, R_ee, o_ee.copy(), 4))

    return dict(
        b=b, Rb=Rb, eul=eul,
        joint_axes=[a1, a2, a3, a4],
        joint_orig=[o_j1, o_j2, o_j3, o_j4],
        ee_pos=o_ee, ee_R=R_ee,
        bodies=bodies,
    )


def body_jacobians(kin):
    """Linear (Jv) and angular (Jw) Jacobians (3x10) for every body of the robot,
    expressed w.r.t. that robot's 10 generalized coordinates.  Analytic."""
    b = kin["b"]
    eul = kin["eul"]
    W = W_euler(eul)
    axes = kin["joint_axes"]
    orig = kin["joint_orig"]

    jacs = []
    for (_name, _m, _I, _R, com, jmax) in kin["bodies"]:
        Jv = np.zeros((3, 10))
        Jw = np.zeros((3, 10))
        # base translation
        Jv[:, 0:3] = np.eye(3)
        # base rotation (Euler rates -> body omega via W); linear coupling v = omega x r
        r = com - b
        Jw[:, 3:6] = W
        Jv[:, 3:6] = -skew(r) @ W
        # arm joints that move this body
        for i in range(jmax):
            ai = axes[i]
            Jw[:, 6 + i] = ai
            Jv[:, 6 + i] = np.cross(ai, com - orig[i])
        jacs.append((Jv, Jw))
    return jacs


# ----------------------------------------------------------------------------
# Generalized inertia D(q), gravity G(q)
# ----------------------------------------------------------------------------
def robot_inertia_gravity(qr):
    """10x10 inertia block and 10-vector gravity for a single robot."""
    kin = robot_kinematics(qr)
    jacs = body_jacobians(kin)
    Dr = np.zeros((10, 10))
    Gr = np.zeros(10)
    for (_n, m, Il, Rw, com, _j), (Jv, Jw) in zip(kin["bodies"], jacs):
        Iw = Rw @ Il @ Rw.T
        Dr += m * (Jv.T @ Jv) + Jw.T @ Iw @ Jw
        Gr += m * g * Jv[2, :]              # dP/dq = sum m g dz_com/dq
    return Dr, Gr


def payload_inertia(pe):
    """6x6 payload inertia block from its ZYX-Euler angles."""
    Wp = W_euler(pe)
    Dp = np.zeros((6, 6))
    Dp[0:3, 0:3] = M_PAY * np.eye(3)
    Dp[3:6, 3:6] = Wp.T @ I_PAY @ Wp
    return Dp


def inertia_and_gravity(q):
    D = np.zeros((NQ, NQ))
    G = np.zeros(NQ)

    # robots
    for R in ROBOTS:
        sl = R["all"]
        Dr, Gr = robot_inertia_gravity(q[sl])
        D[sl, sl] = Dr
        G[sl] = Gr

    # payload: free rigid body
    D[PAY["all"], PAY["all"]] = payload_inertia(q[PAY["eul"]])
    G[22] = M_PAY * g                       # only pz carries payload gravity

    # tiny regularization for numerical SPD safety
    D += 1e-9 * np.eye(NQ)
    return D, G


def inertia(q):
    return inertia_and_gravity(q)[0]


def gravity(q):
    return inertia_and_gravity(q)[1]


# Actuation matrix: 8 arm torques act on the 8 arm coordinates.
B = np.zeros((NQ, 8))
arm_cols = list(range(6, 10)) + list(range(16, 20))
for j, c in enumerate(arm_cols):
    B[c, j] = 1.0


# ----------------------------------------------------------------------------
# Coriolis term C(q,qd) qd via finite-difference Christoffel symbols.
#   C qd = Ddot qd - 1/2 grad_q( qd^T D qd )
# ----------------------------------------------------------------------------
def _block_coriolis(Dfun, qb, qdb, eps=1e-6):
    """C qd for one block via  Ddot qd - 1/2 grad(qd^T D qd), D = Dfun(qb)."""
    nb = qb.size
    D0 = Dfun(qb)
    Ddot_qd = ((Dfun(qb + eps * qdb) - D0) / eps) @ qdb
    s0 = qdb @ D0 @ qdb
    grad = np.empty(nb)
    for k in range(nb):
        db = qb.copy()
        db[k] += eps
        grad[k] = (qdb @ Dfun(db) @ qdb - s0) / eps
    return Ddot_qd - 0.5 * grad


def coriolis(q, qd, eps=1e-6):
    """C(q,qd) qd via the Christoffel identity, exploiting the block-diagonal
    structure of D: each robot block and the payload block are independent, so a
    perturbation of one robot's coordinates only rebuilds that 10-DOF block."""
    out = np.zeros(NQ)
    for R in ROBOTS:
        sl = R["all"]
        out[sl] = _block_coriolis(lambda qb: robot_inertia_gravity(qb)[0],
                                  q[sl], qd[sl], eps)
    # payload: translational inertia is constant (no Coriolis); only the
    # rotational 3-DOF Euler sub-block W^T I W is configuration dependent.
    def D_pay_eul(pe):
        Wp = W_euler(pe)
        return Wp.T @ I_PAY @ Wp
    out[PAY["eul"]] = _block_coriolis(D_pay_eul, q[PAY["eul"]], qd[PAY["eul"]], eps)
    return out


# ----------------------------------------------------------------------------
# Holonomic contact constraints: each end-effector pinned to its payload frame.
#   kappa_r = [ p_ee_r - p_cf_r ;  vee(Log(R_cf_r^T R_ee_r)) ]   in R^6
# ----------------------------------------------------------------------------
def contact_frame(q, r_idx):
    """World pose (position, rotation) of contact frame r_idx on the payload."""
    p = q[PAY["pos"]]
    Ro = euler_zyx_to_R(q[PAY["eul"]])
    d = CONTACT_OFFSET[r_idx]
    p_cf = p + Ro @ d
    # pad frame nominally aligned with payload frame
    R_cf = Ro
    return p_cf, R_cf


def constraints(q):
    kap = np.zeros(12)
    for i, R in enumerate(ROBOTS):
        kin = robot_kinematics(q[R["all"]])
        p_cf, R_cf = contact_frame(q, i)
        kap[6 * i:6 * i + 3] = kin["ee_pos"] - p_cf
        kap[6 * i + 3:6 * i + 6] = log_SO3(R_cf.T @ kin["ee_R"])
    return kap


def constraint_jacobian(q, eps=1e-6):
    J = np.zeros((12, NQ))
    k0 = constraints(q)
    for k in range(NQ):
        dq = np.zeros(NQ)
        dq[k] = eps
        J[:, k] = (constraints(q + dq) - k0) / eps
    return J, k0


# ----------------------------------------------------------------------------
# Output: payload-pose tracking error  y = y_a(q) - y_d(t)   in R^6
# ----------------------------------------------------------------------------
def payload_pose(q):
    return q[PAY["pos"]].copy(), euler_zyx_to_R(q[PAY["eul"]])


def output(q, traj, t):
    pd, Rd, _, _, _ = traj(t)
    p, Ro = payload_pose(q)
    y = np.zeros(6)
    y[0:3] = p - pd
    y[3:6] = log_SO3(Rd.T @ Ro)
    return y


def output_jacobian(q, traj, t, eps=1e-6):
    """J_y = d y / d q  (6x26) at fixed time t.  The orientation rows depend on
    the desired rotation R_d(t) through y = (p - p_d, Log(R_d^T R_o)), so the
    trajectory and time are passed in and held constant during differentiation."""
    J = np.zeros((6, NQ))
    y0 = output(q, traj, t)
    for k in range(NQ):
        dq = np.zeros(NQ)
        dq[k] = eps
        J[:, k] = (output(q + dq, traj, t) - y0) / eps
    return J
