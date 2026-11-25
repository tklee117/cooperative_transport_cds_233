"""
CLF-QP controller for the 26-DOF cooperative pinch-lift-move system.

Builds on the same constraint-consistent relative-degree-2 output model used by
the feedback-linearizing controller (../feedback_linearization).  The output
dynamics are

    yddot = L_f^2 y_eff(x) + Ghat(x) u ,         Ghat in R^{6x8}

obtained from the KKT-reduced constrained dynamics  qddot = alpha0 + Mu u .

With eta = (y, ydot) in R^12 and the chain-of-integrators form
    etadot = F eta + G yddot ,   F = [[0,I6],[0,0]],  G = [[0],[I6]],
we take the CLF
    V(eta) = eta^T P eta ,
where P solves the Lyapunov equation for the *desired* feedback-linearized
closed loop A_cl = [[0,I],[-Kp,-Kd]]:
    A_cl^T P + P A_cl = -Q .
Then
    Vdot = LfV + LgV u ,
    LfV  = eta^T (F^T P + P F) eta + 2 eta^T P G L_f^2 y_eff ,
    LgV  = 2 eta^T P G Ghat   (1x8) .
The feedback-linearizing input mu = -K eta gives Vdot = -eta^T Q eta = -r V with
r = lambda_min(Q)/lambda_max(P), so the CLF decrease condition Vdot <= -r V is
feasible and the FL law is one admissible member.

The CLF-QP keeps only this single scalar decrease constraint (vs the 6 equality
constraints implicit in exact FL), freeing 7 input DOF -- which it directs toward
an arm-posture objective u_des instead of pinning them by the pseudo-inverse:

    u = argmin_u  ||u - u_des||^2   s.t.   LgV u <= -r V - LfV .

For a single inequality with an identity-weighted cost this QP has a closed form
(projection onto the half-space), so no external QP solver is needed.
"""
import os
import sys

import numpy as np
from scipy.linalg import solve_lyapunov, solve_continuous_are

# reuse the validated model / controller machinery
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "feedback_linearization"))
import model as M          # noqa: E402
import controller as C     # noqa: E402

ARM_IDX = C.ARM_IDX

# chain-of-integrators output model (relative degree 2, k = 6)
F12 = np.zeros((12, 12)); F12[:6, 6:] = np.eye(6)
G12 = np.zeros((12, 6)); G12[6:, :] = np.eye(6)


def clf_from_gains(Kp, Kd, Q=None):
    """Lyapunov CLF P and rate r for the desired FL closed loop A_cl."""
    A_cl = np.block([[np.zeros((6, 6)), np.eye(6)], [-Kp, -Kd]])
    if Q is None:
        Q = np.eye(12)
    P = solve_lyapunov(A_cl.T, -Q)          # A_cl^T P + P A_cl = -Q
    r = np.linalg.eigvalsh(Q).min() / np.linalg.eigvalsh(P).max()
    return P, r, Q


def clf_from_care(omega_o):
    """Well-conditioned CLF for the double-integrator output via the algebraic
    Riccati equation.  With Q_lqr = diag(omega_o^4 I, 2 omega_o^2 I), R = I the
    LQR-optimal closed loop is critically damped at s = -omega_o (the same poles
    as the FL A_cl), but the Riccati P is far better conditioned than the
    Lyapunov P, so V -> 0 actually drives ||y|| -> 0 over realistic horizons.
    Returns (P, r_nat) with r_nat = lambda_min(Q+K^T R K)/lambda_max(P)."""
    R = np.eye(6)
    Qc = np.block([[omega_o**4 * np.eye(6), np.zeros((6, 6))],
                   [np.zeros((6, 6)), 2 * omega_o**2 * np.eye(6)]])
    P = solve_continuous_are(F12, G12, Qc, R)
    K = np.linalg.solve(R, G12.T @ P)        # optimal gain
    Qcl = Qc + K.T @ R @ K
    r = np.linalg.eigvalsh(Qcl).min() / np.linalg.eigvalsh(P).max()
    return P, r


def clf_qp_control(q, qd, traj, t, P, r, baumgarte=(40.0, 1.0), ridge=1e-2,
                   extra_force=None, posture=None, relax=None):
    """One CLF-QP control step.

    posture : optional (kp_n, kd_n, arm_nom) -> u_des is an arm posture-PD torque
              that the QP honours wherever the output CLF constraint leaves room.
              If None, u_des = 0 (minimum-norm CLF-QP).
    relax   : optional penalty weight for a soft constraint (slack); None -> hard.
    """
    alpha0, Mu, info = C.kkt_solve(q, qd, traj, t, baumgarte, extra_force=extra_force)
    y = M.output(q, traj, t)
    ydot, Jy, _ = C.output_rate(q, qd, traj, t)

    # L_f^2 y_eff = output acceleration under u=0 = directional derivative of ydot
    # along the u=0 flow (q,qd) -> (qd, alpha0).  Computed as a clean central
    # difference along the flow -- accurate even when qd != 0, unlike the nested
    # spatial finite difference (FL tolerates that error because it cancels the
    # same estimate exactly; the CLF-QP constraint does not, so accuracy matters).
    h = 1e-6
    ydp, _, _ = C.output_rate(q + h * qd, qd + h * alpha0, traj, t + h)
    ydm, _, _ = C.output_rate(q - h * qd, qd - h * alpha0, traj, t - h)
    Lf2 = (ydp - ydm) / (2 * h)          # 6

    Ghat = Jy @ Mu                       # 6x8
    eta = np.concatenate([y, ydot])      # 12

    PG = P @ G12                          # 12x6
    LfV = eta @ (F12.T @ P + P @ F12) @ eta + 2.0 * eta @ PG @ Lf2
    LgV = 2.0 * eta @ PG @ Ghat          # 1x8
    Vval = eta @ P @ eta
    b = -r * Vval - LfV                  # constraint:  LgV . u <= b

    if posture is not None:
        kp_n, kd_n, arm_nom = posture
        u_des = -kp_n * (q[ARM_IDX] - arm_nom) - kd_n * qd[ARM_IDX]
    else:
        u_des = np.zeros(8)

    a = LgV
    aa = a @ a
    viol = a @ u_des - b
    if aa < 1e-12 or viol <= 0.0:
        u = u_des                                    # constraint inactive / no authority
    else:
        u = u_des - (viol / aa) * a                  # project onto half-space boundary

    sigma_min = info_sigma(Ghat)
    diag = dict(y=y, ydot=ydot, V=Vval, Vdot=LfV + LgV @ u, rV=r * Vval,
                sigma_min=sigma_min, active=float(viol > 0.0))
    return u, alpha0, Mu, info, diag


def info_sigma(Ghat):
    try:
        return np.linalg.svd(Ghat, compute_uv=False)[-1] if np.all(np.isfinite(Ghat)) else np.nan
    except np.linalg.LinAlgError:
        return np.nan
