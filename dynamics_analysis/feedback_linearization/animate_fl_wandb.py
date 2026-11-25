"""
Render the feedback-linearizing (FL) controller's behavior on the 26-DOF
cooperative pinch-lift-move system as an MP4 and upload it to Weights & Biases.

This visualizes *how the FL controller tracks the commanded payload trajectory*:
the two Go2 bases + D1 4-DOF arms pinch the payload and the constraint-consistent
FL law drives the payload along p_d(t) (a smooth lift-and-move), while the
internal arm/base configuration evolves under the (non-minimum-phase) zero
dynamics. Both the actual payload and the commanded ghost are drawn so the
tracking is visible frame by frame.

Everything is the SAME analytic plant/controller used by study.py -- this script
only adds rendering + wandb logging. It does NOT use Isaac Lab (the FL controller
is not wired into the Isaac sim; see README).

Run:
    python animate_fl_wandb.py                 # default lift-and-move, logs to wandb
    python animate_fl_wandb.py --no-wandb      # just write the mp4 locally
    python animate_fl_wandb.py --T 0.8 --dt 1e-3 --name fl-liftmove
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import model as M
import controller as C
import ic as IC
import study as S   # reuse the exact validated rk4 rollout + settings

OUT = os.path.dirname(os.path.abspath(__file__))

# 12 edges of a unit box given its 8 corners (corner ordering below)
_BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0),
              (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(center, R, extents):
    """World-frame 8 corners of a box at pose (center, R) with full extents."""
    hx, hy, hz = np.asarray(extents) / 2.0
    local = np.array([[sx * hx, sy * hy, sz * hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (center[None, :] + local @ R.T)


def box_segments(center, R, extents):
    c = box_corners(center, R, extents)
    return [(c[i], c[j]) for i, j in _BOX_EDGES]


def arm_polyline(qr):
    """Ordered world points base -> joints -> end-effector for one robot."""
    kin = M.robot_kinematics(qr)
    pts = [kin["b"]] + list(kin["joint_orig"]) + [kin["ee_pos"]]
    return np.array(pts), kin["ee_pos"]


def trim_record(rec, y_max=0.25, kap_max=0.5):
    """Cut the record at the first diverged step so a blown-up tail never reaches
    the video or the wandb scalars. The FL output is non-minimum phase, so a long
    enough rollout always eventually drifts; we keep only the valid window."""
    Q, Y, KAP = rec["Q"], rec["Y"], rec["KAP"]
    n = len(rec["t"])
    nv = n
    for i in range(n):
        bad = (not np.all(np.isfinite(Q[i]))
               or not np.all(np.isfinite(Y[i]))
               or np.linalg.norm(Y[i]) > y_max
               or KAP[i] > kap_max)
        if bad:
            nv = i
            break
    if nv < n:
        print(f"[trim] valid window = {nv}/{n} steps (t={rec['t'][nv-1]:.3f}s); "
              f"dropping diverged tail")
    return {k: v[:nv] for k, v in rec.items()}


def run_rollout(args):
    """FL-controlled lift-and-move rollout; returns the study.py record dict and
    the commanded trajectory so we can draw the ghost payload."""
    q_nom = IC.project_config(IC.nominal_config())[0]
    p0 = q_nom[M.PAY["pos"]].copy()
    p_goal = p0 + np.array(args.move)          # lift + translate command
    traj = C.make_trajectory(p0, p_goal, T_move=args.t_move)

    # initial condition: optionally start the payload offset from nominal and let
    # the FL law regulate it back (Experiment-A regime, the validated working
    # case). Project the offset IC back onto the pinch constraint manifold.
    q0 = q_nom.copy()
    off = np.array(args.ic_offset)
    if np.any(off != 0.0):
        q0[M.PAY["pos"]] += off[:3]
        q0[M.PAY["eul"]] += off[3:]
        q0, res = IC.project_config(q0)
        print(f"[ic] payload offset {off}  ->  ||kappa0||={res:.1e}")

    # Low-level locomotion inner loop on the 12 floating-base DOF.  study.passive
    # uses kp_base=0 (free base) to EXPOSE the zero-dynamics instability.  Here the
    # inner loop is active, and -- crucially for TRANSPORT -- it is a *locomotion*
    # policy, not a static posture hold: each base walks horizontally to follow the
    # payload it carries, keeping the D1 arms near their well-conditioned nominal
    # shape.  Without this the arms over-extend into a singularity within ~0.1 s
    # (the failure mode documented in the report); with it the arms only have to
    # supply the lift + fine pose tracking, so the FL law can transport the payload.
    pay_nom = q_nom[M.PAY["pos"]].copy()
    fg = args.follow_gain
    kp_arm = args.kp_arm
    arm_idx = list(range(6, 10)) + list(range(16, 20))
    arm_nom = q_nom[arm_idx].copy()

    def locomotion_passive(q, qd, q_nom_, base_support=True):
        qset = q_nom_.copy()
        if fg != 0.0:
            # horizontal displacement of the payload from its start -> walk bases
            dxy = q[M.PAY["pos"]][:2] - pay_nom[:2]
            qset[0:2] += fg * dxy        # robot-1 base (x,y) follows payload
            qset[10:12] += fg * dxy      # robot-2 base (x,y) follows payload
        tau = C.passive_force(q, qd, qset, base_support=True,
                              b_arm=args.b_arm, kp_base=args.kp_base, kd_base=args.kd_base)
        if kp_arm != 0.0:
            # null-space arm-posture spring toward nominal shape: this force is
            # supplied to BOTH controller and plant, so the FL law cancels it in the
            # 6 output directions -- it only damps the (unstable) internal arm mode,
            # buying enough stable horizon to complete a finite transport.
            tau[arm_idx] += -kp_arm * (q[arm_idx] - arm_nom)
        return tau

    S.passive = locomotion_passive

    # honor CLI dt/T by overriding study's module-level knobs the rollout reads
    S.DT = args.dt
    print(f"[rollout] T={args.T}s dt={args.dt}s  move={args.move}  T_move={args.t_move}")
    t0 = time.time()
    rec = S.rk4_rollout(q0, np.zeros(M.NQ), traj, q_nom,
                        T=args.T, dt=args.dt, base_support=True, control=True)
    print(f"[rollout] {len(rec['t'])} steps in {time.time()-t0:.0f}s  "
          f"||y0||={np.linalg.norm(rec['Y'][0]):.3e}  "
          f"||y(T)||={np.linalg.norm(rec['Y'][-1]):.3e}")
    return rec, traj, q_nom


def render(rec, traj, q_nom, args):
    t = rec["t"]
    Q = rec["Q"]
    nstep = len(t)
    stride = max(1, nstep // args.frames)
    idx = list(range(0, nstep, stride))

    # commanded payload position history (for the trailing ghost path)
    pd_hist = np.array([traj(t[i])[0] for i in range(nstep)])

    # scene bounds from all drawn geometry (ignore any non-finite rows from a
    # diverged tail so a late NaN can't break the whole render)
    pts = np.vstack([Q[:, M.PAY["pos"]], pd_hist,
                     Q[:, M.R1["base_pos"]], Q[:, M.R2["base_pos"]]])
    pts = pts[np.isfinite(pts).all(1)]
    lo = pts.min(0) - 0.35
    hi = pts.max(0) + 0.35
    ctr = 0.5 * (lo + hi)
    rng = max(hi - lo) / 2.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame_i):
        ax.cla()
        i = idx[frame_i]
        q = Q[i]

        # --- ground grid ---
        gx = np.linspace(ctr[0] - rng, ctr[0] + rng, 7)
        gy = np.linspace(ctr[1] - rng, ctr[1] + rng, 7)
        for xv in gx:
            ax.plot([xv, xv], [gy[0], gy[-1]], [0, 0], color="0.85", lw=0.6)
        for yv in gy:
            ax.plot([gx[0], gx[-1]], [yv, yv], [0, 0], color="0.85", lw=0.6)

        # --- commanded (desired) payload: ghost box + full planned path ---
        pd, Rd, _, _, _ = traj(t[i])
        segs = box_segments(pd, Rd, M.BOX)
        ax.add_collection3d(Line3DCollection(segs, colors="tab:orange",
                                             linewidths=1.2, linestyles="--", alpha=0.7))
        # draw the ENTIRE desired trajectory (faint) so the goal is visible up front,
        # plus the traversed portion (bold) so progress along it reads frame by frame
        ax.plot(pd_hist[:, 0], pd_hist[:, 1], pd_hist[:, 2],
                color="tab:orange", lw=1.0, ls=":", alpha=0.45)
        ax.plot(pd_hist[:i + 1, 0], pd_hist[:i + 1, 1], pd_hist[:i + 1, 2],
                color="tab:orange", lw=2.0, alpha=0.9, label="desired payload trajectory")
        ax.scatter(*pd_hist[0], color="tab:orange", marker="o", s=35)   # start
        ax.scatter(*pd_hist[-1], color="tab:orange", marker="*", s=130)  # goal

        # --- actual payload box + traversed path ---
        p, Ro = M.payload_pose(q)
        segs = box_segments(p, Ro, M.BOX)
        ax.add_collection3d(Line3DCollection(segs, colors="tab:blue", linewidths=2.0))
        ax.plot(Q[:i + 1, 20], Q[:i + 1, 21], Q[:i + 1, 22],
                color="tab:blue", lw=2.0, alpha=0.9, label="actual payload (FL-tracked)")
        # leader line from actual payload center to its commanded target (the error)
        ax.plot([p[0], pd[0]], [p[1], pd[1]], [p[2], pd[2]],
                color="0.4", lw=1.0, ls="-", alpha=0.7)

        # --- robots: base marker + arm polyline + EE pad ---
        for R, col, lab in ((M.R1, "tab:green", "robot 1 (base walks + arm)"),
                            (M.R2, "tab:red", "robot 2 (base walks + arm)")):
            poly, ee = arm_polyline(q[R["all"]])
            ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color=col, lw=2.5, marker="o", ms=3,
                    label=lab)
            ax.scatter(*q[R["base_pos"]], color=col, s=60, marker="s")  # base
            ax.scatter(*ee, color=col, s=40, marker="^")                # contact pad

        ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
        ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
        ax.set_zlim(0, ctr[2] + rng)
        ax.set_box_aspect((1, 1, 0.7))
        ax.view_init(elev=18, azim=-60)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

        pos_err = np.linalg.norm(rec["Y"][i][:3])
        ori_err = np.linalg.norm(rec["Y"][i][3:])
        sig = rec["SIG"][i]
        mode = "FL transport + base locomotion" if args.follow_gain else "FL output tracking"
        ax.set_title(
            f"{mode}   t={t[i]:.3f}s\n"
            f"payload pos err={pos_err*1e3:.1f} mm   ori err={ori_err*1e3:.1f} mrad   "
            f"$\\sigma_{{\\min}}(\\hat G)$={sig:.3f}",
            fontsize=10)
        return []

    fps = args.fps
    print(f"[render] {len(idx)} frames @ {fps} fps  (~{len(idx)/fps:.1f}s video)")
    anim = FuncAnimation(fig, draw, frames=len(idx), blit=False)
    mp4 = os.path.join(OUT, f"{args.name}.mp4")
    anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    print(f"[render] wrote {mp4}")
    return mp4, idx, pd_hist


def log_wandb(mp4, rec, idx, args):
    import wandb
    wandb.init(project=args.project, name=args.name, config={
        "system": "26-DOF cooperative pinch-lift-move (analytic, no IsaacLab)",
        "controller": "constraint-consistent feedback linearization",
        "omega_o": S.OMEGA_O, "dt": args.dt, "T": args.T,
        "move_command": list(args.move), "t_move": args.t_move,
        "follow_gain": args.follow_gain, "kp_base": args.kp_base, "kd_base": args.kd_base,
        "base_locomotion": args.follow_gain != 0.0,
    })
    t = rec["t"]
    p0 = rec["Q"][0][20:23].copy()              # payload start (for transport progress)
    b1_0 = rec["Q"][0][0:2].copy(); b2_0 = rec["Q"][0][10:12].copy()
    # controller / tracking time series (the 'how it tracks' scalars)
    for i in range(len(t)):
        y = rec["Y"][i]
        arm = np.concatenate([rec["Q"][i][6:10], rec["Q"][i][16:20]])
        arm0 = np.concatenate([rec["Q"][0][6:10], rec["Q"][0][16:20]])
        p = rec["Q"][i][20:23]
        # how well each base kept its formation offset under the payload (the
        # locomotion job): drift of base->payload horizontal vector from t=0
        f1 = np.linalg.norm((rec["Q"][i][0:2] - p[:2]) - (b1_0 - p0[:2]))
        f2 = np.linalg.norm((rec["Q"][i][10:12] - p[:2]) - (b2_0 - p0[:2]))
        wandb.log({
            "t": float(t[i]),
            "track/payload_pos_err_mm": float(np.linalg.norm(y[:3]) * 1e3),
            "track/payload_ori_err_mrad": float(np.linalg.norm(y[3:]) * 1e3),
            "track/output_norm": float(np.linalg.norm(y)),
            "transport/payload_disp_m": float(np.linalg.norm(p - p0)),
            "transport/payload_x": float(p[0]),
            "transport/payload_z": float(p[2]),
            "locomotion/base_formation_drift": float(max(f1, f2)),
            "controller/sigma_min_Ghat": float(rec["SIG"][i]),
            "controller/fl_residual": float(rec["RES"][i]) if np.isfinite(rec["RES"][i]) else None,
            "constraint/kappa_norm": float(rec["KAP"][i]),
            "internal/arm_shape_drift": float(np.linalg.norm(arm - arm0)),
            "internal/base_z1": float(rec["Q"][i][2]),
            "internal/base_z2": float(rec["Q"][i][12]),
        }, step=i)
    wandb.log({"fl_rollout_video": wandb.Video(mp4, fps=args.fps, format="mp4")})
    wandb.finish()
    print(f"[wandb] uploaded video + tracking scalars to project '{args.project}' run '{args.name}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=0.8, help="rollout horizon [s]")
    ap.add_argument("--dt", type=float, default=1e-3, help="integration step [s]")
    ap.add_argument("--t_move", type=float, default=0.5, help="trajectory move time [s]")
    ap.add_argument("--move", type=float, nargs=3, default=[0.12, 0.0, 0.10],
                    help="payload lift+translate command [dx dy dz] (m)")
    ap.add_argument("--ic_offset", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0],
                    help="initial payload offset [dx dy dz droll dpitch dyaw] to regulate from")
    ap.add_argument("--kp_base", type=float, default=250.0, help="base posture P gain (inner loop)")
    ap.add_argument("--kd_base", type=float, default=60.0, help="base posture D gain (inner loop)")
    ap.add_argument("--follow_gain", type=float, default=0.0,
                    help="locomotion gain: fraction of payload horizontal motion the "
                         "bases walk to follow (1.0 = bases track payload x,y; 0 = static hold)")
    ap.add_argument("--b_arm", type=float, default=5.0, help="arm joint viscous damping")
    ap.add_argument("--kp_arm", type=float, default=0.0,
                    help="null-space arm-posture spring toward nominal shape (damps the "
                         "unstable internal mode; cancelled in the 6 output directions)")
    ap.add_argument("--frames", type=int, default=120, help="target #animation frames")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--name", type=str, default="fl-pinch-lift-move")
    ap.add_argument("--project", type=str, default="decplm-fl")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    rec, traj, q_nom = run_rollout(args)
    rec = trim_record(rec)
    if len(rec["t"]) < 5:
        raise SystemExit("[error] rollout diverged almost immediately; "
                         "try smaller --move, larger --kp_base, or smaller --dt")
    mp4, idx, pd_hist = render(rec, traj, q_nom, args)
    if not args.no_wandb:
        log_wandb(mp4, rec, idx, args)


if __name__ == "__main__":
    main()
