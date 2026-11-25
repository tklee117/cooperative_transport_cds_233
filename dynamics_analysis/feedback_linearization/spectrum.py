"""
Rigorous zero-dynamics assessment by linearization (complements the rollout in
study.py / Experiment B).  Two findings, both produced here:

  (1) NO EQUILIBRIUM ON Z.  Holding the (unactuated) payload requires arm torques
      that necessarily induce internal acceleration; with only a 2-D control
      redundancy this internal acceleration cannot be nulled.  We show that the
      smallest internal acceleration achievable over the 8-DOF contact-consistent
      self-motion manifold is bounded away from zero (~20 rad/s^2), even with full
      gravity compensation.  Hence the zero dynamics have no rest point and the
      configuration DRIFTS -- the dominant mechanism behind "drift to
      singularities".

  (2) VARIATIONAL SPECTRUM.  Linearizing the closed loop about the nominal point
      on Z, A = d f_cl/d x, the 12 output-error modes sit at the placed location
      s = -omega_o (with a Baumgarte constraint cluster nearby), while several
      internal/zero-dynamics modes have Re(lambda) > 0 -- so the zero dynamics are
      also locally expansive, not merely drifting.

Run:  python spectrum.py   -> writes fig_C_eigenvalue_spectrum.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model as M
import controller as C
import ic as IC
import analysis as A

OUT = os.path.dirname(os.path.abspath(__file__))

OMEGA_O = 40.0
KP = OMEGA_O**2 * np.eye(6)
KD = 2 * OMEGA_O * np.eye(6)
BAUMGARTE = (40.0, 1.0)
RIDGE = 1e-2
B_ARM = 5.0
KD_BASE = 15.0


def main():
    q_nom = IC.project_config(IC.nominal_config())[0]
    p0 = q_nom[M.PAY["pos"]].copy()
    traj = C.make_trajectory(p0, p0, T_move=1.0)        # hold payload -> motion on Z

    # (1) no-equilibrium check: minimal internal acceleration over the self-motion
    #     manifold, with and without arm gravity compensation.
    print("(1) equilibrium feasibility on Z (min achievable ||q̈||):")
    for tag, gc in (("bare FL", False), ("FL + arm gravity comp", True)):
        f = A.make_closed_loop(traj, KP, KD, q_nom, baumgarte=BAUMGARTE, ridge=RIDGE,
                               base_support=True, b_arm=B_ARM, kd_base=KD_BASE,
                               grav_comp_arm=gc)
        _, res = A.find_equilibrium(f, q_nom, iters=40)
        a0 = np.linalg.norm(f(np.concatenate([q_nom, np.zeros(M.NQ)]))[M.NQ:])
        print(f"    {tag:24s}: ||q̈|| at nominal = {a0:7.2f},  min over self-motion = {res:7.2f}  "
              f"({'NO equilibrium' if res > 1.0 else 'equilibrium found'})")

    # (2) variational spectrum about the nominal point on Z (bare FL)
    f = A.make_closed_loop(traj, KP, KD, q_nom, baumgarte=BAUMGARTE, ridge=RIDGE,
                           base_support=True, b_arm=B_ARM, kd_base=KD_BASE,
                           grav_comp_arm=False)
    x0 = np.concatenate([q_nom, np.zeros(M.NQ)])
    drift = np.linalg.norm(f(x0)[M.NQ:])
    print(f"\n(2) variational linearization about nominal point on Z "
          f"(residual drift ||q̈||={drift:.1f} rad/s²):")
    Amat, _ = A.linearize(f, x0)
    w = A.spectrum_report(Amat)
    re = w.real
    n_unstable = int((re > 1e-2).sum())
    print(f"    max Re(λ) = {re.max():.3f},  # modes with Re>1e-2 = {n_unstable}")
    print("    unstable eigenvalues:", np.round(w[re > 1e-2], 3))
    print(f"    placed output-error poles expected at s = -{OMEGA_O:.0f}")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    unst = re > 1e-2
    near_pole = (np.abs(w.real + OMEGA_O) < 6.0) & (~unst)
    other_stable = (~unst) & (~near_pole)
    for sl, c, lab, mk in [
        (other_stable, "tab:blue", "stable internal / zero-dynamics modes", "o"),
        (near_pole, "0.6", r"placed output + Baumgarte poles ($\approx -\omega_o$)", "s"),
        (unst, "tab:red", "UNSTABLE zero-dynamics modes", "X"),
    ]:
        if sl.any():
            ax[0].scatter(w[sl].real, w[sl].imag, c=c, marker=mk, s=55,
                          edgecolors="k", linewidths=.4, label=lab, zorder=3)
    ax[0].axvline(0, color="k", lw=.8); ax[0].axvline(-OMEGA_O, color="0.6", lw=.8, ls=":")
    ax[0].set(title="(a) closed-loop variational spectrum (52 eigenvalues)",
              xlabel="Re(λ)  [1/s]", ylabel="Im(λ)  [1/s]")
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=.3)

    for sl, c, mk in [(other_stable, "tab:blue", "o"), (unst, "tab:red", "X")]:
        if sl.any():
            ax[1].scatter(w[sl].real, w[sl].imag, c=c, marker=mk, s=75,
                          edgecolors="k", linewidths=.4, zorder=3)
    ax[1].axvline(0, color="k", lw=.8)
    ax[1].set_xlim(-12, max(2.0, re.max() * 1.4))
    ax[1].set(title=f"(b) zoom near jω axis — {n_unstable} unstable modes, max Re(λ)={re.max():.2f}/s",
              xlabel="Re(λ)  [1/s]", ylabel="Im(λ)  [1/s]")
    ax[1].grid(alpha=.3)

    fig.suptitle("Zero-dynamics linearization: no equilibrium on Z (drift) + unstable internal modes")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_C_eigenvalue_spectrum.png", dpi=130)
    print("  wrote fig_C_eigenvalue_spectrum.png")


if __name__ == "__main__":
    main()
