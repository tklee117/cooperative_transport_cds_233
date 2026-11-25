"""
Rigorous zero-dynamics assessment by linearization.

Idea: under exact feedback linearization the 12 output-error coordinates
eta=(y,ydot) obey eta_dot = A_cl eta (Hurwitz by design).  The remaining 40
states are the zero-dynamics coordinates.  We

  1. find a closed-loop equilibrium x* = (q*, 0) with f_cl(x*) = 0,
  2. numerically linearize the closed loop, A = d f_cl / d x | x*,
  3. report the spectrum.  Eigenvalues that are NOT the placed output-error
     poles are the zero-dynamics modes; max Re(.) > 0 => unstable zero dynamics.

This does not rely on the nonlinear rollout staying bounded, so it gives a clean
verdict even when the bare controller's zero dynamics blow up.
"""
import numpy as np
import model as M
import controller as C
import ic as IC


def make_closed_loop(traj, Kp, Kd, q_nom, baumgarte=(15.0, 1.0), ridge=1e-2,
                     base_support=True, b_arm=2.0, kd_base=15.0, nullspace=None,
                     grav_comp_arm=False):
    """Return f_cl(x,t) -> xdot for the full constrained closed loop."""
    def f_cl(x, t=0.0):
        q, qd = x[:M.NQ], x[M.NQ:]
        pf = C.passive_force(q, qd, q_nom, base_support=base_support,
                             b_arm=b_arm, kp_base=0.0, kd_base=kd_base,
                             grav_comp_arm=grav_comp_arm)
        u, a0, Mu, info, d = C.fbl_control(q, qd, traj, t, Kp, Kd,
                                           baumgarte=baumgarte, ridge=ridge,
                                           extra_force=pf, nullspace=nullspace)
        return np.concatenate([qd, a0 + Mu @ u])
    return f_cl


def find_equilibrium(f_cl, q0, iters=60, tol=1e-9, damp=1e-6):
    """Solve for q* (with qd=0) such that the acceleration part of f_cl([q*,0])
    vanishes, using a robust trust-region least-squares.  Returns (q*, residual).

    A real equilibrium on the zero-dynamics surface must keep the holonomic
    contact constraints satisfied, so each candidate q is first projected onto
    {kappa=0}; we therefore parametrise the search by the 8 internal/redundant
    DOF (the contact-consistent self-motion) and let the projection set the rest.
    """
    from scipy.optimize import least_squares

    # internal coordinates we are free to move (arm joints + base x,y,yaw of each
    # robot); the projection then fixes the remaining base DOF to keep kappa=0.
    free = [6, 7, 8, 9, 0, 1, 5, 16, 17, 18, 19, 10, 11, 15]

    def residual(p):
        q = q0.copy()
        q[free] = p
        q, _ = IC.project_config(q)
        return f_cl(np.concatenate([q, np.zeros(M.NQ)]))[M.NQ:]

    sol = least_squares(residual, q0[free], method="lm", max_nfev=iters * len(free),
                        xtol=1e-12, ftol=1e-12)
    q = q0.copy(); q[free] = sol.x
    q, _ = IC.project_config(q)
    return q, np.linalg.norm(f_cl(np.concatenate([q, np.zeros(M.NQ)]))[M.NQ:])


def linearize(f_cl, x_eq, eps=1e-6):
    """Finite-difference Jacobian A = d f_cl / d x at x_eq (52x52)."""
    nx = x_eq.size
    f0 = f_cl(x_eq)
    A = np.zeros((nx, nx))
    for k in range(nx):
        dx = np.zeros(nx); dx[k] = eps
        A[:, k] = (f_cl(x_eq + dx) - f_cl(x_eq - dx)) / (2 * eps)
    return A, f0


def spectrum_report(A):
    w = np.linalg.eigvals(A)
    w = w[np.argsort(-w.real)]
    return w
