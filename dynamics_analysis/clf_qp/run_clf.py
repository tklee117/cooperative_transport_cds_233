"""
Head-to-head simulation of the CLF-QP controller against the feedback-linearizing
(FL) law on the 26-DOF constrained pinch-lift-move system.

Compares, per the report's task list:
  (1) output-error convergence rate,
  (2) null-space drift rate (arm-shape divergence on the zero-dynamics surface),
  (3) total control effort.

Controllers:
  FL              : u = Ghat^+(-Lf2 + v)                 (exact decoupling)
  CLF-QP (min)    : min ||u||^2  s.t. Vdot <= -rV         (single scalar constraint)
  CLF-QP (posture): min ||u - u_posture||^2 s.t. Vdot<=-rV (uses freed DOF for posture)

Run:  python run_clf.py [--quick]   -> writes fig_clf_comparison.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "feedback_linearization"))
import model as M          # noqa: E402
import controller as C     # noqa: E402
import ic as IC            # noqa: E402
import clf_qp as Q         # noqa: E402

QUICK = "--quick" in sys.argv
OUT = os.path.dirname(os.path.abspath(__file__))

OMEGA_O = 40.0
KP = OMEGA_O**2 * np.eye(6)
KD = 2 * OMEGA_O * np.eye(6)
BAUMGARTE = (40.0, 1.0)
RIDGE = 1e-2
B_ARM = 5.0
KD_BASE = 15.0
DT = 0.001 if QUICK else 0.0005

# Well-conditioned CLF V = eta^T P eta from the CARE (same -omega_o poles as the
# FL closed loop, but conditioned so V->0 actually drives ||y||->0).  The CLF
# *decrease rate* enforced by the QP is a design parameter; we pick it below
# omega_o so the QP keeps slack to direct toward the arm-posture objective.
P, R_NAT = Q.clf_from_care(OMEGA_O)
R_CLF = 30.0
print(f"CLF (CARE): cond(P)={np.linalg.cond(P):.1f}, natural rate {R_NAT:.2f}/s; "
      f"chosen design rate r = {R_CLF:.1f}/s")

ARM = Q.ARM_IDX


def passive(q, qd, q_nom):
    return C.passive_force(q, qd, q_nom, base_support=True, b_arm=B_ARM,
                           kp_base=0.0, kd_base=KD_BASE)


def make_controllers(q_nom):
    arm_nom = q_nom[ARM]
    post = (60.0, 15.0, arm_nom)

    def fl(q, qd, traj, t, pf):
        u, a0, Mu, info, d = C.fbl_control(q, qd, traj, t, KP, KD,
                                           baumgarte=BAUMGARTE, ridge=RIDGE, extra_force=pf)
        return u, a0, Mu, d

    def clf_min(q, qd, traj, t, pf):
        u, a0, Mu, info, d = Q.clf_qp_control(q, qd, traj, t, P, R_CLF,
                                              baumgarte=BAUMGARTE, ridge=RIDGE,
                                              extra_force=pf, posture=None)
        return u, a0, Mu, d

    def clf_post(q, qd, traj, t, pf):
        u, a0, Mu, info, d = Q.clf_qp_control(q, qd, traj, t, P, R_CLF,
                                              baumgarte=BAUMGARTE, ridge=RIDGE,
                                              extra_force=pf, posture=post)
        return u, a0, Mu, d

    return {"FL": fl, "CLF-QP (min)": clf_min, "CLF-QP (posture)": clf_post}


def rollout(ctrl, q0, qd0, traj, q_nom, T, dt=DT):
    n = int(round(T / dt))

    def rhs(x, t, log=None):
        q, qd = x[:M.NQ], x[M.NQ:]
        u, a0, Mu, d = ctrl(q, qd, traj, t, passive(q, qd, q_nom))
        if log is not None:
            log.update(u=u, **d)
        return np.concatenate([qd, a0 + Mu @ u])

    x = np.concatenate([q0, qd0])
    rec = dict(t=[], Q=[], Y=[], U=[])
    for i in range(n + 1):
        t = i * dt
        log = {}
        k1 = rhs(x, t, log)
        rec["t"].append(t); rec["Q"].append(x[:M.NQ].copy())
        rec["Y"].append(log["y"]); rec["U"].append(log["u"])
        if i == n or not np.all(np.isfinite(x)):
            break
        k2 = rhs(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    for k in rec:
        rec[k] = np.array(rec[k])
    return rec


def arm_drift(rec, q_nom):
    return np.linalg.norm(rec["Q"][:, ARM] - q_nom[ARM], axis=1)


def effort(rec):
    un = np.linalg.norm(rec["U"], axis=1)
    return np.trapz(un, rec["t"])           # integral_0^T ||u|| dt  [N*m*s]


def fit_rate(t, x, t0):
    m = (t > t0) & (x > 0) & np.isfinite(x)
    if m.sum() < 5:
        return float("nan")
    return np.polyfit(t[m], np.log(x[m]), 1)[0]


# ----------------------------------------------------------------------------
def main():
    q_nom = IC.project_config(IC.nominal_config())[0]
    p0 = q_nom[M.PAY["pos"]].copy()
    hold = C.make_trajectory(p0, p0, T_move=1.0)
    ctrls = make_controllers(q_nom)

    # ---- Experiment 1: zero-dynamics IC (y0 = 0, internal DOF perturbed) ----
    T1 = 0.25 if QUICK else 0.4
    qzd = q_nom.copy()
    qzd[6:10] += np.array([0.04, -0.03, 0.03, -0.02])
    qzd[16:20] += np.array([-0.03, 0.03, -0.03, 0.02])
    qzd, _ = IC.project_config(qzd)
    print(f"\n[zero-dynamics IC]  y0 = {np.linalg.norm(M.output(qzd, hold, 0.0)):.2e}")
    exp1 = {}
    for name, ctrl in ctrls.items():
        t0 = time.time()
        rec = rollout(ctrl, qzd, np.zeros(M.NQ), hold, q_nom, T1)
        exp1[name] = rec
        d = arm_drift(rec, q_nom)
        print(f"  {name:18s}: drift {d[0]:.3f}->{d[-1]:.3f} (rate {fit_rate(rec['t'], d, 0.05):+.1f}/s), "
              f"effort {effort(rec):.2f} N·m·s, ||y(T)||={np.linalg.norm(rec['Y'][-1]):.2e} ({time.time()-t0:.0f}s)")

    # ---- Experiment 2: output IC (payload perturbed in controllable x-z) ----
    T2 = 0.15 if QUICK else 0.2
    qoc = q_nom.copy()
    qoc[M.PAY["pos"]] += np.array([0.025, 0.0, 0.02])
    qoc, _ = IC.project_config(qoc)
    print(f"\n[output IC]  y0 = {np.linalg.norm(M.output(qoc, hold, 0.0)):.3f}")
    exp2 = {}
    for name in ("FL", "CLF-QP (min)", "CLF-QP (posture)"):
        t0 = time.time()
        rec = rollout(ctrls[name], qoc, np.zeros(M.NQ), hold, q_nom, T2)
        exp2[name] = rec
        ny = np.linalg.norm(rec["Y"], axis=1)
        print(f"  {name:18s}: ||y|| {ny[0]:.3f}->{ny[-1]:.2e} (rate {fit_rate(rec['t'], ny, 0.0):+.1f}/s), "
              f"effort {effort(rec):.2f} N·m·s ({time.time()-t0:.0f}s)")

    plot(exp1, exp2, q_nom)


def plot(exp1, exp2, q_nom):
    col = {"FL": "tab:red", "CLF-QP (min)": "tab:orange", "CLF-QP (posture)": "tab:green"}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) Exp1 arm-shape drift (the null-space / zero-dynamics metric)
    for name, rec in exp1.items():
        ax[0, 0].semilogy(rec["t"], np.maximum(arm_drift(rec, q_nom), 1e-6),
                          color=col[name], label=name)
    ax[0, 0].set(title="(a) zero-dynamics IC: arm-shape drift ||Δθ_arm||",
                 xlabel="t [s]", ylabel="rad"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(True, which="both", alpha=.3)

    # (b) Exp1 output excursion
    for name, rec in exp1.items():
        ax[0, 1].semilogy(rec["t"], np.maximum(np.linalg.norm(rec["Y"], axis=1), 1e-12),
                          color=col[name], label=name)
    ax[0, 1].set(title="(b) zero-dynamics IC: output excursion ||y||",
                 xlabel="t [s]"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(True, which="both", alpha=.3)

    # (c) Exp1 cumulative control effort
    for name, rec in exp1.items():
        cum = np.array([np.trapz(np.linalg.norm(rec["U"][:i + 1], axis=1), rec["t"][:i + 1])
                        for i in range(len(rec["t"]))])
        ax[1, 0].plot(rec["t"], cum, color=col[name], label=name)
    ax[1, 0].set(title=r"(c) cumulative control effort $\int_0^t\|u\|\,d\tau$",
                 xlabel="t [s]", ylabel="N·m·s"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)

    # (d) Exp2 output convergence (FL vs CLF-QP variants)
    for name, rec in exp2.items():
        ax[1, 1].semilogy(rec["t"], np.linalg.norm(rec["Y"], axis=1),
                          color=col[name], label=name)
    ax[1, 1].set(title="(d) output IC: convergence ||y(t)|| (FL vs CLF-QP)",
                 xlabel="t [s]"); ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(True, which="both", alpha=.3)

    fig.suptitle("CLF-QP vs feedback linearization on the 26-DOF constrained system")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_clf_comparison.png", dpi=130)
    print("  wrote fig_clf_comparison.png")


if __name__ == "__main__":
    main()
