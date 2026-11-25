# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Standalone (no Isaac Sim) validation of the coupled box + 2-agent CBF safety filter.

This is the dependency-light companion to the in-sim experiment. It models the decPLM grasp
constellation kinematically -- a box plus two robot bases rigidly offset along the box x-axis -- and
drives it with a constant *box-frame* twist command, exactly the signal decPLM tracks at play time.

It runs two rollouts through the same obstacle field:

* **No-CBF (baseline):** the nominal command is executed directly. The constellation plows through
  the obstacles (some barrier ``h(t)`` goes negative -> collision), reproducing the open-space
  policy's failure mode.
* **CBF:** the same nominal command is filtered by :func:`cbf_safety_filter`. The QP deflects the box
  twist just enough to keep the box *and both agents* outside every keep-out disk (``h(t) >= 0``),
  intervening only near the obstacles.

It writes a figure (top-down trajectories, barrier ``h(t)``, and intervention magnitude) so you can
eyeball the coupled avoidance before launching the heavy Isaac experiment. Run::

    python cbf_standalone_demo.py --out /tmp/cbf_demo.png

Because the constellation is rigid here, this isolates the *planning-layer* safety guarantee; the
Isaac experiment then tests whether the learned decentralized policy can actually realize the
deflected trajectory while maintaining the grasp.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Import the dependency-free CBF core directly from the sibling mdp/safety package without requiring
# the full extension to be installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "mdp", "safety"))
from cbf_qp import cbf_safety_filter, rotation_2d  # noqa: E402


def _guidance(p, v_box_xy_world, obstacles, obstacle_r, box_r, margin, goal, circ_gain, circ_dist):
    """Goal-seeking velocity toward ``goal`` plus a tangential go-around term near each obstacle.

    Mirrors ``CBFSafeVelocityHeightCommand._guidance_command`` / ``._circulation`` so the standalone
    behaviour matches the in-sim filter."""
    v = goal - p  # (1,2)
    speed_cap = v_box_xy_world  # scalar max speed
    spd = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v = v * (speed_cap / spd).clamp(max=1.0)
    if circ_gain > 0.0:
        d = p[:, None, :] - obstacles  # (1,K,2)
        dist = d.norm(dim=-1).clamp_min(1e-6)
        n = d / dist[..., None]
        t = torch.stack([-n[..., 1], n[..., 0]], dim=-1)
        s = torch.where((t * v[:, None, :]).sum(-1) >= 0, 1.0, -1.0)
        r_eff = box_r + obstacle_r + margin
        act = torch.exp(-(dist - r_eff[None]).clamp_min(0.0) / circ_dist)
        v = v + ((circ_gain * act * s)[..., None] * t).sum(1) * v.norm(dim=-1, keepdim=True)
    return v


def rollout(
    obstacles: torch.Tensor,
    obstacle_r: torch.Tensor,
    use_cbf: bool,
    *,
    goal=None,
    speed: float = 0.4,
    circ_gain: float = 1.2,
    circ_dist: float = 0.6,
    agent_off: float = 0.9,
    body_r=(0.33, 0.35, 0.35),
    alpha: float = 2.0,
    margin: float = 0.05,
    dt: float = 0.05,
    steps: int = 700,
):
    """Kinematically integrate the rigid {box + 2 agents} constellation under goal-seeking guidance,
    optionally filtered by the coupled CBF. Returns stacked time series for plotting."""
    device = "cpu"
    body_r_t = torch.tensor(body_r, device=device)
    local = torch.tensor([[0.0, 0.0], [-agent_off, 0.0], [agent_off, 0.0]], device=device)
    goal = torch.tensor([[goal[0], goal[1]]], device=device) if goal is not None else None

    p = torch.zeros(1, 2, device=device)
    yaw = torch.zeros(1, device=device)
    box_traj, agent0_traj, agent1_traj = [], [], []
    min_h_series, interv_series = [], []

    for _ in range(steps):
        R = rotation_2d(yaw)[0]
        world_bodies = (p[0] + local @ R.T).unsqueeze(0)  # (1, 3, 2)

        # goal-seeking guidance -> body-frame nominal twist. Circulation (go-around) is part of the
        # safety layer, so it is applied ONLY with the CBF -- the baseline goal-seeks straight into
        # obstacles (no obstacle awareness), matching the in-sim CBFSafeVelocityHeightCommand.
        v_world = _guidance(p, speed, obstacles[0], obstacle_r, body_r_t[0], margin, goal,
                            circ_gain if use_cbf else 0.0, circ_dist)
        u_nom = torch.zeros(1, 3, device=device)
        u_nom[:, :2] = R.T @ v_world[0]

        u_safe, diag = cbf_safety_filter(
            u_nom, p, yaw, world_bodies, body_r_t, obstacles, obstacle_r, alpha=alpha, margin=margin
        )
        if not use_cbf:
            u_safe = u_nom

        box_traj.append(world_bodies[0, 0].clone())
        agent0_traj.append(world_bodies[0, 1].clone())
        agent1_traj.append(world_bodies[0, 2].clone())
        min_h_series.append(diag.min_h.item())
        interv_series.append((u_safe - u_nom).norm().item())

        v_exec = R @ u_safe[0, :2]
        p = p + dt * v_exec.unsqueeze(0)
        yaw = yaw + dt * u_safe[0, 2]

    return {
        "box": torch.stack(box_traj).numpy(),
        "agent0": torch.stack(agent0_traj).numpy(),
        "agent1": torch.stack(agent1_traj).numpy(),
        "min_h": torch.tensor(min_h_series).numpy(),
        "intervention": torch.tensor(interv_series).numpy(),
        "dt": dt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="/tmp/cbf_demo.png", help="Output figure path.")
    parser.add_argument("--alpha", type=float, default=2.0, help="CBF class-K gain.")
    parser.add_argument("--margin", type=float, default=0.05, help="Extra safety buffer (m).")
    parser.add_argument("--vel_x", type=float, default=0.4, help="Nominal box-frame v_x (m/s).")
    args = parser.parse_args()

    # A slalom (matching the 2-robot obstacle env): obstacles alternate sides near the centerline.
    # The box goal-seeks toward GOAL past the last one, so it must weave past every obstacle.
    obstacles = torch.tensor([[[1.6, 0.45], [2.8, -0.45], [4.0, 0.45]]])  # (1, K, 2)
    obstacle_r = torch.tensor([0.35, 0.35, 0.35])
    goal = (5.2, 0.0)

    base = rollout(obstacles, obstacle_r, use_cbf=False, goal=goal, speed=args.vel_x,
                   alpha=args.alpha, margin=args.margin)
    safe = rollout(obstacles, obstacle_r, use_cbf=True, goal=goal, speed=args.vel_x,
                   alpha=args.alpha, margin=args.margin)

    print(f"[no-CBF ] min barrier over rollout = {base['min_h'].min():+.3f}  "
          f"(< 0 => collision)   final box = {base['box'][-1].round(3).tolist()}")
    print(f"[ CBF   ] min barrier over rollout = {safe['min_h'].min():+.3f}  "
          f"(>= 0 => safe)       final box = {safe['box'][-1].round(3).tolist()}  (goal {goal})")
    print(f"[ CBF   ] peak intervention ||u_safe - u_nom|| = {safe['intervention'].max():.3f} m/s "
          f"(only engages near obstacles)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("[WARN] matplotlib not available; skipping the figure (numbers above still valid).")
        return

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1])

    # keep-out disks
    for k in range(obstacles.shape[1]):
        ox, oy = obstacles[0, k].tolist()
        ax.add_patch(Circle((ox, oy), obstacle_r[k].item(), color="0.55", alpha=0.35, zorder=0))
        ax.add_patch(Circle((ox, oy), obstacle_r[k].item() + 0.30 + args.margin,
                            color="0.55", fill=False, ls=":", lw=1.0, zorder=0))
        ax.plot(ox, oy, "kx", ms=7, zorder=5)

    ax.plot(base["box"][:, 0], base["box"][:, 1], "r--", lw=2, label="box (no CBF)")
    ax.plot(safe["box"][:, 0], safe["box"][:, 1], "b-", lw=2, label="box (CBF)")
    ax.plot(safe["agent0"][:, 0], safe["agent0"][:, 1], "c-", lw=1, alpha=0.8, label="agent 0 (CBF)")
    ax.plot(safe["agent1"][:, 0], safe["agent1"][:, 1], "m-", lw=1, alpha=0.8, label="agent 1 (CBF)")
    # draw the rigid constellation at a few CBF snapshots
    for idx in range(0, len(safe["box"]), 40):
        xs = [safe["agent0"][idx, 0], safe["agent1"][idx, 0]]
        ys = [safe["agent0"][idx, 1], safe["agent1"][idx, 1]]
        ax.plot(xs, ys, "-", color="0.3", lw=0.8, alpha=0.6, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top-down: coupled CBF deflects box + both agents around obstacles")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    t = base["dt"] * torch.arange(len(base["min_h"])).numpy()
    ax_h.axhline(0.0, color="k", lw=0.8)
    ax_h.plot(t, base["min_h"], "r--", lw=1.5, label="no CBF")
    ax_h.plot(t, safe["min_h"], "b-", lw=1.5, label="CBF")
    ax_h.set_ylabel("min barrier  h(t)")
    ax_h.set_title("Safety: h<0 = inside keep-out (collision)")
    ax_h.legend(fontsize=8)
    ax_h.grid(alpha=0.3)

    ax_u.plot(t, safe["intervention"], "b-", lw=1.5)
    ax_u.set_ylabel(r"$\|u_{safe}-u_{nom}\|$")
    ax_u.set_xlabel("time [s]")
    ax_u.set_title("Filter intervention (minimally invasive)")
    ax_u.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[OK] wrote figure -> {args.out}")


if __name__ == "__main__":
    main()
