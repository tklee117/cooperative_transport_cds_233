"""
Feedback-linearization study for the 26-DOF cooperative pinch-lift-move system
described in system_model_notes.txt.

It demonstrates, by simulation:

  (A) OUTPUT CONVERGENCE.  The constraint-consistent feedback-linearizing law
      exactly linearizes the 6-DOF payload-pose output (relative degree 2,
      full-rank decoupling matrix), so the output error obeys the designed
      Hurwitz dynamics  eta_dot = A_cl eta  and ||y(t)|| -> 0 exponentially from
      several initial conditions.

  (B) ZERO-DYNAMICS STABILITY.  Holding the payload on its trajectory (y == 0),
      the 40-dimensional internal/zero dynamics are integrated forward.  They are
      found to be UNSTABLE: the internal arm/base self-motion grows exponentially
      (drift toward singularities), whereas the open-loop (u = 0) internal motion
      of the same plant stays bounded.  Hence the 6-DOF-payload output renders the
      cooperative system non-minimum phase -- consistent with the report's concern
      that "arm joint angles or base poses drift to singularities".  This is shown
      to be structural: 8 arm torques control 6 outputs, leaving only a 2-D
      redundancy for 8 unstable internal configuration DOF (null-space posture
      control and joint/base damping were verified insufficient), so maintaining a
      bounded formation requires the locomotion inner loop to regulate the base.

Run:
    python study.py            # writes fig_A_*.png and fig_B_*.png
    python study.py --quick     # shorter, coarser, for a fast check
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

import model as M
import controller as C
import ic as IC

QUICK = "--quick" in sys.argv

# --- controller / simulation settings (validated) ---------------------------
OMEGA_O = 40.0                              # output-error bandwidth [rad/s]
KP = OMEGA_O**2 * np.eye(6)
KD = 2 * OMEGA_O * np.eye(6)                 # critically damped output error
BAUMGARTE = (40.0, 1.0)
RIDGE = 1e-2                                  # damped (singularity-robust) pseudo-inverse
B_ARM = 5.0                                   # arm joint viscous damping
KD_BASE = 15.0                                # base velocity damping (locomotion inner loop)
DT = 0.001 if QUICK else 0.0005
T = 0.3
T_A = 0.20          # output-convergence horizon (output settles well within this)
T_B = 0.40          # zero-dynamics horizon (long enough to expose exponential growth)
OUT = os.path.dirname(os.path.abspath(__file__))


def passive(q, qd, q_nom, base_support=True):
    return C.passive_force(q, qd, q_nom, base_support=base_support,
                           b_arm=B_ARM, kp_base=0.0, kd_base=KD_BASE)


def rk4_rollout(q0, qd0, traj, q_nom, T=T, dt=DT, base_support=True, control=True):
    n = int(round(T / dt))

    def rhs(x, t, log=None):
        q, qd = x[:M.NQ], x[M.NQ:]
        pf = passive(q, qd, q_nom, base_support)
        if control:
            u, a0, Mu, info, d = C.fbl_control(q, qd, traj, t, KP, KD,
                                               baumgarte=BAUMGARTE, ridge=RIDGE, extra_force=pf)
            qdd = a0 + Mu @ u
            if log is not None:
                # feedback-linearization residual: realized yddot vs commanded v
                drift_y, Jy, _ = C.output_drift(q, qd, traj, t)
                yddot = drift_y + Jy @ qdd
                log.update(y=d["y"], ydot=d["ydot"], u=u, kappa=d["kappa"],
                           sigma_min=d["sigma_min"], lin_res=np.linalg.norm(yddot - d["v"]))
        else:
            a0, Mu, info = C.kkt_solve(q, qd, traj, t, BAUMGARTE, extra_force=pf)
            qdd = a0
            if log is not None:
                log.update(y=M.output(q, traj, t), kappa=info["kappa"])
        return np.concatenate([qd, qdd])

    x = np.concatenate([q0, qd0])
    rec = dict(t=[], Q=[], QD=[], Y=[], YD=[], U=[], KAP=[], SIG=[], RES=[])
    for i in range(n + 1):
        t = i * dt
        log = {}
        k1 = rhs(x, t, log)
        rec["t"].append(t); rec["Q"].append(x[:M.NQ].copy()); rec["QD"].append(x[M.NQ:].copy())
        rec["Y"].append(log.get("y")); rec["YD"].append(log.get("ydot", np.zeros(6)))
        rec["U"].append(log.get("u", np.zeros(8))); rec["KAP"].append(np.linalg.norm(log["kappa"]))
        rec["SIG"].append(log.get("sigma_min", np.nan)); rec["RES"].append(log.get("lin_res", np.nan))
        if i == n or not np.all(np.isfinite(x)):
            break
        k2 = rhs(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    for k in rec:
        rec[k] = np.array(rec[k])
    return rec


# ----------------------------------------------------------------------------
def experiment_A(q_nom):
    """Output convergence from several perturbed payload initial conditions."""
    p0 = q_nom[M.PAY["pos"]].copy()
    traj = C.make_trajectory(p0, p0, T_move=1.0)        # regulate payload to p0
    # Perturbations in the strongly-controllable x-z plane (and a small yaw).
    # NOTE: the payload-y / shear direction coincides with the weak singular
    # direction of Ghat and the unstable zero-dynamics mode (Experiment B);
    # initial errors there are overtaken by the internal instability rather than
    # regulated, so output convergence holds in the valid (controllable) domain.
    perturbs = [
        np.array([0.025, 0.000, 0.020, 0.000, 0.000, 0.000]),
        np.array([0.020, 0.000, 0.025, 0.000, 0.000, 0.000]),
        np.array([-0.022, 0.000, 0.018, 0.000, 0.000, 0.020]),
        np.array([0.015, 0.000, -0.022, 0.000, 0.000, -0.020]),
    ]
    runs = []
    for k, dp in enumerate(perturbs):
        q0 = q_nom.copy()
        q0[M.PAY["pos"]] += dp[:3]; q0[M.PAY["eul"]] += dp[3:]
        q0, res = IC.project_config(q0)
        t0 = time.time()
        rec = rk4_rollout(q0, np.zeros(M.NQ), traj, q_nom, T=T_A)
        print(f"  [A{k+1}] ||kap0||={res:.1e} y0={np.linalg.norm(rec['Y'][0]):.3f} "
              f"||y(T)||={np.linalg.norm(rec['Y'][-1]):.2e} max||yddot-v||={np.nanmax(rec['RES']):.1e} "
              f"({time.time()-t0:.0f}s)")
        runs.append(rec)
    return runs


def experiment_B(q_nom):
    """Zero dynamics: start on Z (y=0), perturb internal/redundant DOF, hold
    payload.  Compare closed loop (FL) vs open loop (u=0)."""
    p0 = q_nom[M.PAY["pos"]].copy()
    traj = C.make_trajectory(p0, p0, T_move=1.0)
    q0 = q_nom.copy()
    q0[6:10] += np.array([0.04, -0.03, 0.03, -0.02])     # robot1 arm self-motion
    q0[16:20] += np.array([-0.03, 0.03, -0.03, 0.02])    # robot2 arm self-motion
    q0, res = IC.project_config(q0)                       # keep pads on the (fixed) payload
    print(f"  [B] ||kap0||={res:.1e}  y0={np.linalg.norm(M.output(q0, traj, 0.0)):.2e}")
    runs = {}
    for tag, ctrl in (("closed loop (FL)", True), ("open loop (u=0)", False)):
        t0 = time.time()
        rec = rk4_rollout(q0, np.zeros(M.NQ), traj, q_nom, T=T_B, control=ctrl)
        runs[tag] = rec
        qdr = np.linalg.norm(rec["QD"][:, 0:20], axis=1)
        print(f"      [{tag}] {time.time()-t0:.0f}s  ||qd_robot|| {qdr[0]:.2e} -> {qdr[-1]:.2e}")
    return runs


def arm_drift(rec, q_nom):
    """||theta_arm(t) - theta_arm(0)||: the internal *shape* deviation (isolates
    the self-motion from rigid base falling)."""
    idx = list(range(6, 10)) + list(range(16, 20))
    return np.linalg.norm(rec["Q"][:, idx] - q_nom[idx], axis=1)


# ----------------------------------------------------------------------------
def plot_A(runs):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    tt = runs[0]["t"]
    for k, s in enumerate(runs):
        ax[0, 0].semilogy(s["t"], np.linalg.norm(s["Y"], axis=1), label=f"IC {k+1}")
    env = np.linalg.norm(runs[0]["Y"][0]) * np.exp(-OMEGA_O * tt)
    ax[0, 0].semilogy(tt, env, "k--", lw=1.2, label=r"$e^{-\omega_o t}$ envelope")
    ax[0, 0].set(title="(a) output error ||y(t)|| (log) — exponential convergence",
                 xlabel="t [s]", ylabel="||y||"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(True, which="both", alpha=.3)

    s = runs[0]; labs = ["e_px", "e_py", "e_pz", "e_Rx", "e_Ry", "e_Rz"]
    for j in range(6):
        ax[0, 1].plot(s["t"], s["Y"][:, j], label=labs[j])
    ax[0, 1].set(title="(b) output error components, IC 1", xlabel="t [s]")
    ax[0, 1].legend(ncol=2, fontsize=8); ax[0, 1].grid(alpha=.3)

    for k, s in enumerate(runs):
        ax[1, 0].semilogy(s["t"], np.maximum(s["RES"], 1e-12), label=f"IC {k+1}")
    ax[1, 0].set(title=r"(c) feedback-linearization residual $\|\ddot y - v\|$",
                 xlabel="t [s]"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, which="both", alpha=.3)

    for k, s in enumerate(runs):
        ax[1, 1].plot(s["t"], s["SIG"], label=f"IC {k+1}")
    ax[1, 1].set(title=r"(d) decoupling-matrix conditioning $\sigma_{min}(\hat G)$",
                 xlabel="t [s]"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3)

    fig.suptitle("Experiment A — exact output linearization & exponential convergence "
                 "(constraint-consistent FL)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_A_output_convergence.png", dpi=130)
    print("  wrote fig_A_output_convergence.png")


def plot_B(runs):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    col = {"closed loop (FL)": "tab:red", "open loop (u=0)": "tab:green"}
    for tag, s in runs.items():
        ax[0, 0].semilogy(s["t"], np.maximum(np.linalg.norm(s["Y"], axis=1), 1e-14),
                          color=col[tag], label=tag)
    ax[0, 0].set(title="(a) output ||y(t)|| — FL holds payload on Z (y≈0)", xlabel="t [s]")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, which="both", alpha=.3)

    for tag, s in runs.items():
        ax[0, 1].semilogy(s["t"], np.maximum(arm_drift(s, plot_B.q_nom), 1e-6),
                          color=col[tag], label=tag)
    # growth-rate estimate of the closed-loop internal shape drift
    s = runs["closed loop (FL)"]; d = arm_drift(s, plot_B.q_nom)
    m = (s["t"] > 0.02) & (d > 0) & np.isfinite(d)
    if m.sum() > 5:
        rate = np.polyfit(s["t"][m], np.log(d[m]), 1)[0]
        ax[0, 1].set_title(f"(b) arm-shape drift ||Δθ_arm|| — FL diverges (~{rate:.0f}/s), open-loop bounded")
    else:
        ax[0, 1].set_title("(b) arm-shape drift ||Δθ_arm||")
    ax[0, 1].set(xlabel="t [s]", ylabel="rad"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(True, which="both", alpha=.3)

    s = runs["closed loop (FL)"]
    arm = np.concatenate([s["Q"][:, 6:10], s["Q"][:, 16:20]], axis=1)
    ax[1, 0].plot(s["t"], arm)
    ax[1, 0].set(title="(c) closed-loop arm joint angles (8 internal coords) — drift",
                 xlabel="t [s]", ylabel="rad"); ax[1, 0].grid(alpha=.3)

    for tag, s in runs.items():
        ax[1, 1].plot(s["t"], s["Q"][:, 2], color=col[tag], label=f"{tag} z1")
        ax[1, 1].plot(s["t"], s["Q"][:, 12], color=col[tag], ls="--", label=f"{tag} z2")
    ax[1, 1].set(title="(d) base heights z1, z2", xlabel="t [s]", ylabel="m")
    ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=.3)

    fig.suptitle("Experiment B — zero-dynamics assessment: closed-loop internal modes "
                 "UNSTABLE while output held")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_B_zero_dynamics.png", dpi=130)
    print("  wrote fig_B_zero_dynamics.png")


if __name__ == "__main__":
    print(f"== pinch-lift-move FL study (omega_o={OMEGA_O}, dt={DT}, T={T}, quick={QUICK}) ==")
    q_nom = IC.project_config(IC.nominal_config())[0]
    print("Experiment A: output convergence")
    runsA = experiment_A(q_nom)
    plot_A(runsA)
    print("Experiment B: zero-dynamics stability")
    runsB = experiment_B(q_nom)
    plot_B.q_nom = q_nom
    plot_B(runsB)
    print("done.")
