# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Coupled control-barrier-function (CBF) safety filter for the decPLM box-transport task.

This module is **dependency-free** (pure PyTorch) so that it can be unit-tested and exercised by a
standalone 2-D simulation *without* Isaac Sim, and then reused verbatim inside the Isaac Lab command
term (:mod:`...mdp.commands.cbf_safe_velocity_height_command`).

Problem
-------
decPLM was trained in open space. At play time the high-level planner emits a *box-frame* twist
command ``u_nom = [v_x, v_y, omega_z]`` that the learned policy tracks. We wrap that command in a
minimally-invasive QP safety filter:

.. math::

    u^* = \arg\min_u \tfrac12 \lVert u - u_\text{nom} \rVert^2
    \quad\text{s.t.}\quad
    \nabla h_{b,i}(x)^\top \dot p_b(u) \ge -\alpha\, h_{b,i}(x)\ \ \forall b, i

with one barrier per (rigid body ``b``, obstacle ``i``)

.. math::    h_{b,i}(x) = \lVert p_b - o_i \rVert^2 - r_{b,i}^2 .

What makes this a *coupled* (multi-body) CBF: a single decision variable -- the **box twist** -- moves
three rigid bodies at once: the box center (body 0) and the two robot bases (bodies 1, 2), which under
the rigid contact constellation follow the box twist

.. math::    \dot p_b = R(\psi)\,[v_x, v_y]^\top + \omega_z\,\hat z \times (p_b - p_\text{box}).

So the box's own avoidance constraint and *both agents'* avoidance constraints share, and compete for,
the same control input. The filter therefore has to find a box twist that keeps the whole {box + 2
agents} constellation outside every obstacle simultaneously -- exactly the coupling that makes this
non-trivial.

Solver
------
The QP has 3 variables and a handful of inequality constraints, with a strictly-convex identity
Hessian (the objective is a squared distance). We solve it with **Hildreth's algorithm** -- projected
Gauss-Seidel coordinate ascent on the (non-negativity-constrained) dual -- which needs no matrix
inversion and runs fully batched across parallel environments on-device. A velocity-level (single
integrator) CBF-QP is always feasible (``u`` is unbounded), so no slack variable is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "rotation_2d",
    "build_coupled_cbf_constraints",
    "solve_qp_hildreth",
    "cbf_safety_filter",
    "CBFDiagnostics",
]


def rotation_2d(yaw: torch.Tensor) -> torch.Tensor:
    """Batched 2-D rotation matrices ``R(yaw)``.

    Args:
        yaw: ``(B,)`` yaw angles (rad).

    Returns:
        ``(B, 2, 2)`` rotation matrices mapping body-frame planar vectors to world frame.
    """
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    R = torch.empty(yaw.shape[0], 2, 2, device=yaw.device, dtype=yaw.dtype)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R


def build_coupled_cbf_constraints(
    box_pos_xy: torch.Tensor,
    box_yaw: torch.Tensor,
    body_pos_xy: torch.Tensor,
    body_radius: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius: torch.Tensor,
    alpha: float,
    margin: "float | torch.Tensor" = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Assemble the linear CBF constraints ``G u >= rhs`` for the coupled box+agents system.

    The decision variable ``u = [v_x, v_y, omega_z]`` is the **box-frame** twist. The planar world
    velocity of body ``b`` is ``\dot p_b = A_b u`` with

    .. math::    A_b = \big[\; R(\psi)\ \big|\ \hat z \times (p_b - p_\text{box})\;\big] \in \mathbb R^{2\times 3}.

    Each (body, obstacle) pair yields one row ``g^\top u >= -alpha h`` with
    ``g = 2 A_b^\top (p_b - o)`` and barrier ``h = ||p_b - o||^2 - (r_b + r_o + margin)^2``.

    Args:
        box_pos_xy: ``(B, 2)`` box-center world position.
        box_yaw: ``(B,)`` box yaw (rad).
        body_pos_xy: ``(B, Nb, 2)`` world positions of the rigid bodies. Convention: body 0 is the
            box center (so its lever arm is zero); the remaining bodies are the agents/robot bases.
        body_radius: ``(Nb,)`` or ``(B, Nb)`` effective radius of each body.
        obstacle_xy: ``(B, K, 2)`` world positions of the ``K`` obstacles.
        obstacle_radius: ``(K,)`` or ``(B, K)`` obstacle radii.
        alpha: class-:math:`\mathcal K` gain of the CBF (larger -> later, sharper intervention).
        margin: extra safety buffer added to every radius sum.

    Returns:
        ``G`` of shape ``(B, Nb*K, 3)``, ``rhs`` of shape ``(B, Nb*K)``, and the signed barrier
        values ``h_vals`` of shape ``(B, Nb, K)`` (``h_vals < 0`` means inside the keep-out disk).
    """
    B, Nb, _ = body_pos_xy.shape
    K = obstacle_xy.shape[1]
    device, dtype = body_pos_xy.device, body_pos_xy.dtype

    if body_radius.dim() == 1:
        body_radius = body_radius.unsqueeze(0).expand(B, Nb)
    if obstacle_radius.dim() == 1:
        obstacle_radius = obstacle_radius.unsqueeze(0).expand(B, K)

    # Per-body 2x3 control map A_b. Linear-velocity block is R(yaw) (shared across bodies); the
    # angular block is the lever arm perp(p_b - p_box) -> identically zero for the box center.
    R = rotation_2d(box_yaw)  # (B, 2, 2)
    lever = body_pos_xy - box_pos_xy.unsqueeze(1)  # (B, Nb, 2)
    perp = torch.stack((-lever[..., 1], lever[..., 0]), dim=-1)  # z_hat x lever -> (B, Nb, 2)
    A = torch.empty(B, Nb, 2, 3, device=device, dtype=dtype)
    A[..., :2] = R.unsqueeze(1).expand(B, Nb, 2, 2)
    A[..., 2] = perp

    # Barriers and gradients for every (body, obstacle) pair, fully vectorized.
    # `margin` may be a scalar or a per-env tensor (B,) -> reshape to broadcast over (B, Nb, K).
    m = margin.view(B, 1, 1) if torch.is_tensor(margin) else margin
    d = body_pos_xy.unsqueeze(2) - obstacle_xy.unsqueeze(1)  # (B, Nb, K, 2)
    r_eff = body_radius.unsqueeze(2) + obstacle_radius.unsqueeze(1) + m  # (B, Nb, K)
    h_vals = (d * d).sum(-1) - r_eff * r_eff  # (B, Nb, K)

    # g = 2 * A_b^T d   ->   einsum over the 2-D world axis i
    g = 2.0 * torch.einsum("bnki,bnij->bnkj", d, A)  # (B, Nb, K, 3)
    rhs = -alpha * h_vals  # (B, Nb, K)

    G = g.reshape(B, Nb * K, 3)
    rhs = rhs.reshape(B, Nb * K)
    return G, rhs, h_vals


def solve_qp_hildreth(
    u_nom: torch.Tensor,
    G: torch.Tensor,
    rhs: torch.Tensor,
    n_iter: int = 200,
    omega: float = 1.5,
    eps: float = 1e-9,
) -> torch.Tensor:
    r"""Batched solve of ``min 0.5||u - u_nom||^2  s.t.  G u >= rhs`` via Hildreth's algorithm.

    The dual is ``max_{lambda >= 0} -0.5 lambda^T W lambda + b^T lambda`` with ``W = G G^T`` and
    ``b = rhs - G u_nom``; the primal recovery is ``u = u_nom + G^T lambda``. Degenerate rows
    (``W_ii ~ 0``, i.e. a zero constraint gradient) are inert: their multiplier is pinned to 0.

    The coordinate update uses successive over-relaxation (``omega > 1``). Because ``W = G G^T`` is
    rank-deficient when there are more constraints than control dimensions (the usual CBF case), plain
    Gauss-Seidel (``omega = 1``) has flat dual directions and converges slowly; ``omega ~ 1.5``
    reaches feasibility ~3x faster on adversarial problems and is exact on the benign CBF geometry.

    Args:
        u_nom: ``(B, n)`` nominal (unfiltered) control.
        G: ``(B, M, n)`` constraint matrix.
        rhs: ``(B, M)`` constraint right-hand side.
        n_iter: number of (over-relaxed) Gauss-Seidel sweeps.
        omega: SOR relaxation factor in ``(0, 2)``; ``1.0`` recovers plain Hildreth.
        eps: floor for the diagonal ``W_ii`` used to detect degenerate constraints.

    Returns:
        ``(B, n)`` safe control ``u*``.
    """
    B, M, n = G.shape
    if M == 0:
        return u_nom.clone()

    W = torch.bmm(G, G.transpose(1, 2))  # (B, M, M)
    b = rhs - torch.bmm(G, u_nom.unsqueeze(-1)).squeeze(-1)  # (B, M)
    diag = W.diagonal(dim1=1, dim2=2)  # (B, M)
    degenerate = diag < eps
    inv_diag = torch.where(degenerate, torch.zeros_like(diag), 1.0 / diag.clamp_min(eps))

    lam = torch.zeros(B, M, device=G.device, dtype=G.dtype)
    for _ in range(n_iter):
        for i in range(M):
            # s = sum_{j != i} W_ij lambda_j  (use the latest lambda -> Gauss-Seidel)
            s = torch.einsum("bj,bj->b", W[:, i, :], lam) - W[:, i, i] * lam[:, i]
            lam_target = ((b[:, i] - s) * inv_diag[:, i]).clamp_min(0.0)
            # over-relaxed step, re-projected onto lambda >= 0
            lam[:, i] = (lam[:, i] + omega * (lam_target - lam[:, i])).clamp_min(0.0)
    lam = torch.where(degenerate, torch.zeros_like(lam), lam)

    u = u_nom + torch.bmm(G.transpose(1, 2), lam.unsqueeze(-1)).squeeze(-1)
    return u


@dataclass
class CBFDiagnostics:
    """Per-step diagnostics returned by :func:`cbf_safety_filter` (all batched over envs)."""

    h_vals: torch.Tensor  #: ``(B, Nb, K)`` signed barrier values (``< 0`` => penetrating keep-out disk).
    min_h: torch.Tensor  #: ``(B,)`` worst (smallest) barrier over all bodies/obstacles.
    intervention: torch.Tensor  #: ``(B,)`` ``||u_safe - u_nom||`` magnitude of the filter's correction.
    active: torch.Tensor  #: ``(B,)`` bool, whether the filter measurably changed the command.


def cbf_safety_filter(
    u_nom: torch.Tensor,
    box_pos_xy: torch.Tensor,
    box_yaw: torch.Tensor,
    body_pos_xy: torch.Tensor,
    body_radius: torch.Tensor,
    obstacle_xy: torch.Tensor,
    obstacle_radius: torch.Tensor,
    alpha: float = 2.0,
    margin: float = 0.0,
    n_iter: int = 200,
    active_tol: float = 1e-4,
) -> tuple[torch.Tensor, CBFDiagnostics]:
    """Minimally-invasive coupled-CBF safety filter on the box-frame twist command.

    Convenience wrapper that builds the constraints (:func:`build_coupled_cbf_constraints`), solves
    the QP (:func:`solve_qp_hildreth`), and reports diagnostics. See the module docstring for the
    formulation. Returns ``(u_safe, diagnostics)`` where ``u_safe`` has the same shape as ``u_nom``.
    """
    G, rhs, h_vals = build_coupled_cbf_constraints(
        box_pos_xy=box_pos_xy,
        box_yaw=box_yaw,
        body_pos_xy=body_pos_xy,
        body_radius=body_radius,
        obstacle_xy=obstacle_xy,
        obstacle_radius=obstacle_radius,
        alpha=alpha,
        margin=margin,
    )
    u_safe = solve_qp_hildreth(u_nom, G, rhs, n_iter=n_iter)
    intervention = torch.norm(u_safe - u_nom, dim=-1)
    diag = CBFDiagnostics(
        h_vals=h_vals,
        min_h=h_vals.flatten(1).min(dim=-1).values,
        intervention=intervention,
        active=intervention > active_tol,
    )
    return u_safe, diag


# --------------------------------------------------------------------------------------------------
# Self-test: a head-on approach that the filter must deflect. Run: python cbf_qp.py
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    # Single env. Box at origin heading +x at 0.4 m/s straight at an obstacle 3 m ahead, slightly
    # off the centerline so it sits in the swept path. Two agents sit at +/- x along the box (the
    # grasp constellation), radius 0.3 each. The start is collision-free (forward-invariance test).
    box_pos = torch.tensor([[0.0, 0.0]], device=device)
    box_yaw = torch.tensor([0.0], device=device)
    agent_off = 0.9  # |x| offset of each robot base from box center
    bodies = torch.tensor([[[0.0, 0.0], [-agent_off, 0.0], [agent_off, 0.0]]], device=device)
    body_r = torch.tensor([0.30, 0.30, 0.30], device=device)
    obs = torch.tensor([[[3.0, 0.30]]], device=device)
    obs_r = torch.tensor([0.30], device=device)
    u_nom = torch.tensor([[0.4, 0.0, 0.0]], device=device)

    print("=== sanity: far from obstacle, filter is inactive ===")
    far_obs = torch.tensor([[[20.0, 0.0]]], device=device)
    u_safe, d = cbf_safety_filter(u_nom, box_pos, box_yaw, bodies, body_r, far_obs, obs_r, alpha=2.0)
    print(f"  u_nom={u_nom.tolist()[0]}  u_safe={[round(x,4) for x in u_safe.tolist()[0]]}"
          f"  intervention={d.intervention.item():.2e}  (expect ~0)")
    assert d.intervention.item() < 1e-4

    print("\n=== rollout: integrate the constellation under the filtered command ===")
    dt = 0.05
    p = box_pos.clone()
    yaw = box_yaw.clone()
    min_h_seen = float("inf")
    for step in range(160):
        # rebuild world body positions from current box pose (rigid constellation)
        R = rotation_2d(yaw)[0]
        local = torch.tensor([[0.0, 0.0], [-agent_off, 0.0], [agent_off, 0.0]], device=device)
        world_bodies = (p[0] + (local @ R.T)).unsqueeze(0)
        u_safe, d = cbf_safety_filter(
            u_nom, p, yaw, world_bodies, body_r, obs, obs_r, alpha=2.0, margin=0.05
        )
        min_h_seen = min(min_h_seen, d.min_h.item())
        # integrate box pose with the safe twist (world linear vel = R @ [vx,vy])
        v_world = (R @ u_safe[0, :2])
        p = p + dt * v_world.unsqueeze(0)
        yaw = yaw + dt * u_safe[0, 2]
    print(f"  final box pos = {[round(x,3) for x in p.tolist()[0]]}")
    print(f"  min barrier h over rollout = {min_h_seen:.4f}  (>= 0 means never entered any keep-out disk)")
    # The filter should keep every body strictly outside the keep-out disks for the whole rollout.
    assert min_h_seen > -1e-3, "CBF was violated!"
    print("\nAll self-tests passed.")
