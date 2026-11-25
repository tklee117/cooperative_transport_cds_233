"""Initial-condition construction: a nominal pinching configuration and
projection onto the constraint manifold {kappa(q)=0} and its tangent
{J_kappa qd = 0}."""

import numpy as np
import model as M


def nominal_config(pay_pos=(0.0, 0.0, 0.33)):
    """A symmetric two-robot pinch posture; payload at pay_pos, both pads on the
    +/-x faces.  Returned q is only an initial guess (kappa != 0); call
    project_config to land it exactly on the manifold."""
    q = np.zeros(M.NQ)
    px, py, pz = pay_pos
    # payload
    q[M.PAY["pos"]] = [px, py, pz]
    q[M.PAY["eul"]] = [0.0, 0.0, 0.0]

    # posture chosen well away from the decoupling singularity (sigma_min ~ 0.23)
    arm0 = np.array([0.0, 0.40, -0.625, 0.40])
    # robot 1 : behind -x face, facing +x (yaw 0)
    q[0:3] = [px - 0.62, py, pz - 0.16]
    q[3:6] = [0.0, 0.0, 0.0]
    q[6:10] = arm0
    # robot 2 : behind +x face, facing -x (yaw pi)
    q[10:13] = [px + 0.62, py, pz - 0.16]
    q[13:16] = [0.0, 0.0, np.pi]
    q[16:20] = arm0
    return q


def project_config(q, fix_payload=True, iters=80, tol=1e-10, reg=1e-3):
    """Newton/least-norm projection of the robot coordinates onto kappa(q)=0.
    Payload coordinates are held fixed (they are set by the task)."""
    q = q.copy()
    free = list(range(0, 20))            # both robots' 20 coords
    if not fix_payload:
        free += list(range(20, 26))
    free = np.array(free)
    for _ in range(iters):
        Jk, kap = M.constraint_jacobian(q)
        if np.linalg.norm(kap) < tol:
            break
        Jf = Jk[:, free]
        # damped least-norm step  dq = -Jf^T (Jf Jf^T + reg I)^-1 kap
        dqf = -Jf.T @ np.linalg.solve(Jf @ Jf.T + reg * np.eye(12), kap)
        q[free] += dqf
    return q, np.linalg.norm(M.constraints(q))


def project_velocity(q, qd, fix_payload_vel=True):
    """Project qd onto the constraint tangent {J_kappa qd = 0}.  If
    fix_payload_vel, the payload velocity is treated as given and only robot
    velocities are adjusted."""
    Jk, _ = M.constraint_jacobian(q)
    qd = qd.copy()
    if fix_payload_vel:
        free = np.arange(0, 20)
        resid = Jk @ qd                                  # must be zeroed
        Jf = Jk[:, free]
        dqf = -Jf.T @ np.linalg.solve(Jf @ Jf.T + 1e-9 * np.eye(12), resid)
        qd[free] += dqf
    else:
        P = Jk.T @ np.linalg.solve(Jk @ Jk.T + 1e-9 * np.eye(12), Jk)
        qd = qd - P @ qd
    return qd
