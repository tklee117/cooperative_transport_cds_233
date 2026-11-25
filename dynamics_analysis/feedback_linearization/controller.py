"""
Constraint-consistent feedback-linearizing controller for the 26-DOF
pinch-lift-move system, plus the constrained forward dynamics used to integrate
the true closed loop.

IMPORTANT MODELLING NOTE -- why the report's decoupling matrix needs a correction
---------------------------------------------------------------------------------
The report defines the decoupling matrix as  G = J_y D^{-1} B.  Because the
generalized inertia is block diagonal, the output Jacobian J_y is nonzero only on
the payload columns, while B is nonzero only on the arm-torque rows.  Hence
J_y D^{-1} B is identically ZERO: the arm torques cannot accelerate the payload
*through the inertia matrix alone*.  Physically the payload is unactuated and is
driven purely by the contact wrenches F (the Lagrange multipliers of the
holonomic pinch constraints).  The correct relative-degree-2 input/output map is
therefore obtained through the *constrained* dynamics, in which F = F(q,qd,u):

    [ D   -Jk^T ] [ qdd ]   [ B u - h ]
    [ Jk   0    ] [  F  ] = [  -a_k   ]          (KKT / index-3 -> acceleration form)

Solving this linear system gives  qdd = alpha0 + Mu @ u  (affine in u), so

    ydot = J_y qd + y_t
    yddot = drift_y + J_y qdd = (drift_y + J_y alpha0) + (J_y Mu) u
          = Lf2_eff + Ghat u .

This Ghat (6x8) is the *constraint-consistent* decoupling matrix and is full row
rank away from arm/gimbal singularities, exactly as the report argues for G.
"""

import numpy as np
import model as M


# ----------------------------------------------------------------------------
# Desired payload trajectory  p_d(t):  (p, R) and their 1st/2nd time derivatives.
# Returns (pd, Rd, vd, ad, wd_dummy).  Orientation kept constant here (Rd=I) so
# the lift/move task is a pure translation profile; easy to change.
# ----------------------------------------------------------------------------
def make_trajectory(p0, p_goal, lift_height=0.15, T_move=3.0, hold=None):
    """Smooth min-jerk-ish translation from p0 to p_goal, with a vertical lift
    folded into p_goal[2].  Orientation held at identity."""
    p0 = np.asarray(p0, float)
    p_goal = np.asarray(p_goal, float)
    Rd = np.eye(3)

    def s_curve(t):
        # quintic 0->1 on [0,T], flat afterwards
        if t <= 0:
            return 0.0, 0.0, 0.0
        if t >= T_move:
            return 1.0, 0.0, 0.0
        tau = t / T_move
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T_move
        sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T_move**2
        return s, sd, sdd

    def traj(t):
        s, sd, sdd = s_curve(t)
        pd = p0 + s * (p_goal - p0)
        vd = sd * (p_goal - p0)
        ad = sdd * (p_goal - p0)
        return pd, Rd, vd, ad, np.zeros(3)

    return traj


# ----------------------------------------------------------------------------
# Output rate  ydot = d/dt y(q(t), t)  with q held to first order in qd.
# ----------------------------------------------------------------------------
def output_rate(q, qd, traj, t, ht=1e-6):
    Jy = M.output_jacobian(q, traj, t)
    yt = (M.output(q, traj, t + ht) - M.output(q, traj, t - ht)) / (2 * ht)
    return Jy @ qd + yt, Jy, yt


def output_drift(q, qd, traj, t, eps=1e-6, ht=1e-6):
    """The part of yddot that does NOT contain qdd:
        drift_y = (d ydot / d q) qd + d ydot / dt        (qd held fixed)."""
    yd0, Jy, _ = output_rate(q, qd, traj, t)
    yd_qp, _, _ = output_rate(q + eps * qd, qd, traj, t)
    yd_qm, _, _ = output_rate(q - eps * qd, qd, traj, t)
    term_q = (yd_qp - yd_qm) / (2 * eps)
    yd_tp, _, _ = output_rate(q, qd, traj, t + ht)
    yd_tm, _, _ = output_rate(q, qd, traj, t - ht)
    term_t = (yd_tp - yd_tm) / (2 * ht)
    return term_q + term_t, Jy, yd0


# ----------------------------------------------------------------------------
# Constrained dynamics building blocks
# ----------------------------------------------------------------------------
def constraint_accel_drift(q, qd, eps=1e-6):
    """a_k = Jdot_k qd  (acceleration-level constraint drift)."""
    Jk, kap = M.constraint_jacobian(q)
    Jk_p, _ = M.constraint_jacobian(q + eps * qd)
    Jk_dot = (Jk_p - Jk) / eps
    return Jk_dot @ qd, Jk, kap


def base_support_force(q, qd, q_nom, kp=400.0, kd=80.0):
    """Generalized force supplied by the *frozen low-level locomotion policy*
    (report's modelling assumption: the base behaves as a velocity-controlled
    rigid body held by an inner loop, not an unactuated free body).  It cancels
    gravity on the 12 floating-base DOFs and adds a PD posture hold toward the
    nominal base pose.  It acts ONLY on base coordinates -- never on the arm
    torques (which remain the high-level control u) or the payload."""
    _, G = M.inertia_and_gravity(q)
    tau = np.zeros(M.NQ)
    base_idx = list(range(0, 6)) + list(range(10, 16))
    for k in base_idx:
        tau[k] = G[k] - kp * (q[k] - q_nom[k]) - kd * qd[k]
    return tau


def passive_force(q, qd, q_nom, base_support=True, b_arm=3.0, b_pay=1.0,
                  kp_base=300.0, kd_base=60.0, grav_comp_arm=False):
    """Total *known* generalized force acting on the plant besides the arm
    torques u:
      * joint viscous damping on the 8 arm joints (the D1 actuators carry
        damping 0.5-25 in the repo configs) -- dissipates internal/zero-dynamics
        energy;
      * light translational/rotational damping on the payload;
      * optional base support from the frozen low-level locomotion policy
        (gravity compensation + PD posture hold on the 12 base DOFs);
      * optional arm gravity compensation (grav_comp_arm) -- a computed-torque
        feedforward that removes the static gravitational drift so the nominal
        point becomes a genuine equilibrium on Z (used for the rigorous
        zero-dynamics linearization in spectrum.py).
    This force is supplied to BOTH the controller and the plant, so it is exactly
    cancelled in the 6 output directions and only shapes the zero dynamics."""
    tau = np.zeros(M.NQ)
    arm_idx = list(range(6, 10)) + list(range(16, 20))
    need_G = base_support or grav_comp_arm
    if need_G:
        _, G = M.inertia_and_gravity(q)
    for k in arm_idx:
        tau[k] += -b_arm * qd[k]
        if grav_comp_arm:
            tau[k] += G[k]
    for k in range(20, 26):
        tau[k] += -b_pay * qd[k]
    if base_support:
        for k in list(range(0, 6)) + list(range(10, 16)):
            tau[k] += G[k] - kp_base * (q[k] - q_nom[k]) - kd_base * qd[k]
    return tau


def kkt_solve(q, qd, traj, t, baumgarte=(20.0, 1.0), extra_force=None):
    """Assemble and solve the index-reduced constrained dynamics.

    extra_force : optional known generalized force (e.g. base support) added to
                  the plant, D qdd = B u - h + extra_force + Jk^T F.

    Returns
        alpha0  (26,)   qdd when u = 0
        Mu      (26,8)  d qdd / d u
        info    dict with D, h, Jk, kappa, F0, etc.
    """
    D, G = M.inertia_and_gravity(q)
    h = M.coriolis(q, qd) + G
    if extra_force is not None:
        h = h - extra_force
    a_k, Jk, kap = constraint_accel_drift(q, qd)

    wn, zeta = baumgarte
    cdot = Jk @ qd
    rhs_bot = -(a_k + 2 * zeta * wn * cdot + wn**2 * kap)

    nC = Jk.shape[0]
    K = np.zeros((M.NQ + nC, M.NQ + nC))
    K[:M.NQ, :M.NQ] = D
    K[:M.NQ, M.NQ:] = -Jk.T
    K[M.NQ:, :M.NQ] = Jk
    # constraint-compliance regularization: keeps the KKT system invertible even
    # when a contact Jacobian momentarily loses rank (near arm singularities).
    K[M.NQ:, M.NQ:] = -1e-6 * np.eye(nC)

    # RHS: column 0 -> u=0 part; columns 1..8 -> response to each unit torque
    RHS = np.zeros((M.NQ + nC, 1 + 8))
    RHS[:M.NQ, 0] = -h
    RHS[M.NQ:, 0] = rhs_bot
    RHS[:M.NQ, 1:] = M.B

    try:
        X = np.linalg.solve(K, RHS)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(K, RHS, rcond=None)[0]
    alpha0 = X[:M.NQ, 0]
    Mu = X[:M.NQ, 1:]
    F0 = X[M.NQ:, 0]
    info = dict(D=D, h=h, Jk=Jk, kappa=kap, F0=F0, cdot=cdot)
    return alpha0, Mu, info


# ----------------------------------------------------------------------------
# Feedback-linearizing control law
# ----------------------------------------------------------------------------
ARM_IDX = list(range(6, 10)) + list(range(16, 20))


def fbl_control(q, qd, traj, t, Kp, Kd, baumgarte=(20.0, 1.0), ridge=1e-8,
                extra_force=None, nullspace=None):
    """Constraint-consistent feedback-linearizing control.

    nullspace : optional (kp_n, kd_n, arm_nom).  Because the system is
        over-actuated (8 arm torques, 6 outputs), the 2-D null space of the
        decoupling matrix is used to regulate the arm posture toward arm_nom.
        This stabilizes the otherwise-unstable internal (zero) dynamics WITHOUT
        disturbing the linearized output:  u = Ghat^+ (-Lf2 + v) + (I-Ghat^+ Ghat) u_n.
    """
    alpha0, Mu, info = kkt_solve(q, qd, traj, t, baumgarte, extra_force=extra_force)
    drift_y, Jy, ydot = output_drift(q, qd, traj, t)
    y = M.output(q, traj, t)

    Ghat = Jy @ Mu                       # 6x8 constraint-consistent decoupling matrix
    Lf2_eff = drift_y + Jy @ alpha0      # drift of yddot

    v = -Kp @ y - Kd @ ydot              # exponentially stabilizing PD on (y, ydot)

    # damped (singularity-robust) pseudo-inverse of the non-square Ghat
    GGt = Ghat @ Ghat.T + ridge * np.eye(6)
    Gpinv = Ghat.T @ np.linalg.inv(GGt)
    u = Gpinv @ (-Lf2_eff + v)

    if nullspace is not None:
        kp_n, kd_n, arm_nom = nullspace
        N = np.eye(8) - Gpinv @ Ghat                       # projector onto null(Ghat)
        u_n = -kp_n * (q[ARM_IDX] - arm_nom) - kd_n * qd[ARM_IDX]
        u = u + N @ u_n

    try:
        sigma_min = np.linalg.svd(Ghat, compute_uv=False)[-1] if np.all(np.isfinite(Ghat)) else np.nan
    except np.linalg.LinAlgError:
        sigma_min = np.nan
    diag = dict(y=y, ydot=ydot, v=v, Ghat=Ghat, sigma_min=sigma_min,
                F0=info["F0"], kappa=info["kappa"])
    return u, alpha0, Mu, info, diag


def closed_loop_accel(q, qd, u, alpha0, Mu):
    """qdd of the true constrained system under torque u (reuses the KKT solve)."""
    return alpha0 + Mu @ u
