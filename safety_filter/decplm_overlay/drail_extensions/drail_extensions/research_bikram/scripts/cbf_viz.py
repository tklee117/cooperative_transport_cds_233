# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Visualization helpers for the CBF obstacle-avoidance eval.

Turns a per-step trajectory record (captured in ``play_cbf.py`` straight from the CBF command term)
into (1) a static performance figure and (2) an animated top-down GIF, both ready to log to wandb.
Everything is generic over the number of agents, so it works for the 2-robot and the N-robot tasks.

Record schema (all numpy, env 0, in env-local coordinates):
    box:        (T, 2)    box-center xy trajectory
    yaw:        (T,)      box yaw
    agents:     (T, Na, 2) per-agent base xy trajectory
    height:     (T,)      box height
    min_h:      (T,)      worst CBF barrier over all bodies/obstacles
    intervention:(T,)     ||u_safe - u_nom||
    obstacles:  (K, 2)    obstacle centers
    obstacle_r: (K,)      obstacle radii
    body_r:     (Nb,)     effective radii of [box, agent_0, ...]
    margin:     float     CBF safety margin
    dt:         float     control step
    cbf:        str       "on" | "off"
"""

from __future__ import annotations

import numpy as np

_AGENT_COLORS = ["#1f9e89", "#c0392b", "#8e44ad", "#d68910", "#2471a3", "#d81b60"]


def _reset_indices(box: np.ndarray, thresh: float = 0.4) -> np.ndarray:
    """Indices `i` where the box teleported between step i-1 and i (an env reset on termination).

    The box moves < ~0.01 m per control step at command speeds, so any single-step jump > `thresh`
    (default 0.4 m) is a respawn, not motion. Used to break trajectory lines so the figure doesn't
    draw a straight "elastic" segment across the teleport.
    """
    if box.shape[0] < 2:
        return np.array([], dtype=int)
    step = np.linalg.norm(np.diff(box, axis=0), axis=1)
    return np.where(step > thresh)[0] + 1


def _segments(n: int, resets: np.ndarray):
    """Yield (start, end) index slices split at the reset boundaries."""
    bounds = [0, *resets.tolist(), n]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e > s:
            yield s, e


def _draw_obstacles(ax, rec, inflate_with: float):
    from matplotlib.patches import Circle

    for k in range(rec["obstacles"].shape[0]):
        ox, oy = rec["obstacles"][k]
        r = float(rec["obstacle_r"][k])
        ax.add_patch(Circle((ox, oy), r, color="0.45", alpha=0.45, zorder=1))
        ax.add_patch(Circle((ox, oy), r + inflate_with, color="0.45", fill=False,
                            ls=":", lw=1.0, zorder=1))
        ax.plot(ox, oy, "kx", ms=6, zorder=5)


def make_performance_figure(rec: dict, out_path: str) -> str | None:
    """Static 4-panel figure: top-down trajectories, barrier h(t), intervention, box height."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    T = rec["box"].shape[0]
    Na = rec["agents"].shape[1]
    t = rec["dt"] * np.arange(T)
    box_r = float(rec["body_r"][0])
    agent_r = float(rec["body_r"][1]) if len(rec["body_r"]) > 1 else 0.35

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[2, 1])

    # env resets (terminations) teleport the constellation back to spawn; break lines at those jumps
    resets = _reset_indices(rec["box"])

    _draw_obstacles(ax, rec, inflate_with=box_r + rec["margin"])
    for si, (s, e) in enumerate(_segments(T, resets)):
        ax.plot(rec["box"][s:e, 0], rec["box"][s:e, 1], "-", color="#15397a", lw=2.5,
                label="box" if si == 0 else None, zorder=4)
    ax.plot(rec["box"][0, 0], rec["box"][0, 1], "o", color="#15397a", ms=7, zorder=6)
    ax.plot(rec["box"][-1, 0], rec["box"][-1, 1], "s", color="#15397a", ms=7, zorder=6)
    for a in range(Na):
        c = _AGENT_COLORS[a % len(_AGENT_COLORS)]
        for si, (s, e) in enumerate(_segments(T, resets)):
            ax.plot(rec["agents"][s:e, a, 0], rec["agents"][s:e, a, 1], "-", color=c, lw=1.2,
                    alpha=0.85, label=f"agent {a}" if si == 0 else None, zorder=3)
    # constellation snapshots (box->agent links at a single instant; always valid)
    for idx in range(0, T, max(T // 8, 1)):
        for a in range(Na):
            ax.plot([rec["box"][idx, 0], rec["agents"][idx, a, 0]],
                    [rec["box"][idx, 1], rec["agents"][idx, a, 1]],
                    "-", color="0.6", lw=0.6, alpha=0.5, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    collided = rec["min_h"].min() < 0
    n_reset = len(resets)
    reset_note = f"  |  {n_reset} reset(s)" if n_reset else ""
    ax.set_title(f"Top-down trajectories  (CBF {rec['cbf'].upper()}, {Na} agents)  "
                 f"min h={rec['min_h'].min():+.3f} -> {'COLLISION' if collided else 'SAFE'}{reset_note}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax_h.axhline(0.0, color="k", lw=0.8)
    ax_h.plot(t, rec["min_h"], color="#c0392b" if collided else "#15397a", lw=1.5)
    ax_h.fill_between(t, rec["min_h"], 0, where=rec["min_h"] < 0, color="#c0392b", alpha=0.3)
    for ri in resets:  # mark env resets
        ax_h.axvline(t[ri], color="0.5", ls="--", lw=0.8)
    ax_h.set_ylabel("min barrier h(t)")
    ax_h.set_title("Safety (h<0 = inside keep-out; dashed = reset)")
    ax_h.grid(alpha=0.3)

    ax_u.plot(t, rec["intervention"], color="#117a65", lw=1.5, label="intervention")
    ax_u.set_ylabel(r"$\|u_{safe}-u_{nom}\|$", color="#117a65")
    ax_u.grid(alpha=0.3)
    # overlay the adaptive margin (if it varies) on a twin axis so you can see it auto-tune
    if "margin_eff" in rec and np.ptp(rec["margin_eff"]) > 1e-4:
        ax_m = ax_u.twinx()
        ax_m.plot(t, rec["margin_eff"], color="#b9770e", lw=1.3, ls="--", label="adaptive margin")
        ax_m.set_ylabel("margin [m]", color="#b9770e")
        ax_u.set_title("Filter intervention + adaptive margin")
    else:
        ax_u.set_title("Filter intervention")

    ax_z.plot(t, rec["height"], color="#7d3c98", lw=1.5)
    ax_z.set_ylabel("box height [m]")
    ax_z.set_xlabel("time [s]")
    ax_z.set_title("Lift maintained while deflecting")
    ax_z.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def make_trajectory_gif(rec: dict, out_path: str, fps: int = 20, max_frames: int = 160) -> str | None:
    """Animated top-down GIF of the constellation moving through the obstacle field.

    Bodies turn red on any frame where the worst barrier is negative (a collision), so the difference
    between the baseline and the filtered run is visible at a glance.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        return None

    T = rec["box"].shape[0]
    Na = rec["agents"].shape[1]
    box_r = float(rec["body_r"][0])
    stride = max(T // max_frames, 1)
    frames = list(range(0, T, stride))
    resets = _reset_indices(rec["box"])

    def _seg_start(i):  # index of the start of the (reset-bounded) segment containing i
        prior = resets[resets <= i]
        return int(prior[-1]) if len(prior) else 0

    allx = np.concatenate([rec["box"][:, 0], rec["agents"][:, :, 0].ravel(), rec["obstacles"][:, 0]])
    ally = np.concatenate([rec["box"][:, 1], rec["agents"][:, :, 1].ravel(), rec["obstacles"][:, 1]])
    pad = 0.6
    xlim = (allx.min() - pad, allx.max() + pad)
    ylim = (ally.min() - pad, ally.max() + pad)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    _draw_obstacles(ax, rec, inflate_with=box_r + rec["margin"])

    box_path, = ax.plot([], [], "-", color="#15397a", lw=1.5, alpha=0.6)
    box_dot, = ax.plot([], [], "s", color="#15397a", ms=10)
    agent_dots = [ax.plot([], [], "o", color=_AGENT_COLORS[a % len(_AGENT_COLORS)], ms=8)[0] for a in range(Na)]
    links = [ax.plot([], [], "-", color="0.5", lw=1.0)[0] for _ in range(Na)]
    title = ax.set_title("")

    def update(fi):
        i = frames[fi]
        unsafe = rec["min_h"][i] < 0
        s = _seg_start(i)  # only draw the trail since the last reset (no teleport line)
        box_path.set_data(rec["box"][s:i + 1, 0], rec["box"][s:i + 1, 1])
        box_dot.set_data([rec["box"][i, 0]], [rec["box"][i, 1]])
        box_dot.set_color("#c0392b" if unsafe else "#15397a")
        for a in range(Na):
            ax_, ay_ = rec["agents"][i, a]
            agent_dots[a].set_data([ax_], [ay_])
            links[a].set_data([rec["box"][i, 0], ax_], [rec["box"][i, 1], ay_])
        title.set_text(f"CBF {rec['cbf'].upper()} | t={i * rec['dt']:.1f}s | "
                       f"min h={rec['min_h'][i]:+.2f} {'COLLISION' if unsafe else ''}")
        return [box_path, box_dot, *agent_dots, *links, title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return out_path
