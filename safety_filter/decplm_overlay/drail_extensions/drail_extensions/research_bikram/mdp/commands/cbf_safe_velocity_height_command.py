# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""A box velocity+height command wrapped by a coupled control-barrier-function (CBF) safety filter.

This is a drop-in replacement for :class:`CategoricalVelocityHeightCommand` (the decPLM high-level
"box_command"). Each step, *before the observation manager runs*, it filters the freshly-sampled
box-frame twist command through a small QP so that the box **and both robot bases** stay outside every
configured obstacle. Because the constellation reward and the per-agent belief observations all derive
from ``vel_command_b`` (see :class:`...mdp.observations.transformed_commands_v4`), the learned policy
automatically *observes and tracks the safe command* -- no retraining, identical observation space.

The decision variable is the box twist ``[v_x, v_y, omega_z]``, which under the rigid grasp
constellation moves three bodies at once (box center + two agents), so a single obstacle on one
agent's path bends the whole team's commanded trajectory. See
:mod:`...mdp.safety.cbf_qp` for the formulation and solver.

Set ``enable_cbf=False`` to run the *baseline* (open-space policy, no avoidance) while still logging
the safety diagnostics -- this is the apples-to-apples comparison for "does decPLM break with
obstacles?".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from ..safety.cbf_qp import cbf_safety_filter, rotation_2d
from .categorical_velocity_height_command import (
    CategoricalVelocityHeightCommand,
    CategoricalVelocityHeightCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CBFSafeVelocityHeightCommand(CategoricalVelocityHeightCommand):
    """Categorical velocity+height command with a coupled-CBF obstacle-avoidance safety filter."""

    cfg: CBFSafeVelocityHeightCommandCfg

    def __init__(self, cfg: CBFSafeVelocityHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # The *unfiltered* nominal command (what the high-level planner asks for). We keep it separate
        # from vel_command_b so that filtering does not feed back on itself across steps.
        self.nominal_command_b = torch.zeros(self.num_envs, 4, device=self.device)

        # Obstacle geometry, expressed per-env in the world frame. Obstacles are given in env-local
        # coordinates and shifted by each env's origin so the layout matches the spawned prims.
        obs_local = torch.tensor(
            [[o[0], o[1]] for o in cfg.obstacles], device=self.device, dtype=torch.float32
        )  # (K, 2)
        self._num_obstacles = obs_local.shape[0]
        env_origin_xy = self._env.scene.env_origins[:, :2]  # (B, 2)
        self._obstacle_xy = env_origin_xy.unsqueeze(1) + obs_local.unsqueeze(0)  # (B, K, 2)
        self._obstacle_radius = torch.tensor(
            [o[2] for o in cfg.obstacles], device=self.device, dtype=torch.float32
        )  # (K,)

        # Goal point (env-local) for goal-seeking guidance, shifted per-env by the env origin.
        self._goal_xy = torch.tensor(cfg.goal_xy, device=self.device, dtype=torch.float32)

        # Body (box + agents) effective radii used to inflate every keep-out disk. Supports any
        # number of agents: either an explicit per-body override, or box/agent defaults broadcast.
        self._agent_assets = [self._env.scene[name] for name in cfg.agent_asset_names]
        n_bodies = 1 + len(cfg.agent_asset_names)
        if cfg.body_radii is not None:
            assert len(cfg.body_radii) == n_bodies, (
                f"body_radii has {len(cfg.body_radii)} entries but there are {n_bodies} bodies "
                f"(box + {len(cfg.agent_asset_names)} agents)."
            )
            radii = list(cfg.body_radii)
        else:
            radii = [cfg.box_radius] + [cfg.agent_radius] * len(cfg.agent_asset_names)
        self._body_radius = torch.tensor(radii, device=self.device, dtype=torch.float32)

        # Diagnostics (read by the eval script for safety reporting).
        self.cbf_min_h = torch.zeros(self.num_envs, device=self.device)
        # margin-free physical clearance (m) of the closest body to any obstacle surface; < 0 means an
        # actual overlap, vs cbf_min_h which goes < 0 merely on entering the (inflated) keep-out.
        self.cbf_min_clearance = torch.zeros(self.num_envs, device=self.device)
        self.cbf_intervention = torch.zeros(self.num_envs, device=self.device)
        self.cbf_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cbf_h_vals = torch.zeros(
            self.num_envs, n_bodies, max(self._num_obstacles, 1), device=self.device
        )
        # Per-episode latch: did any body ever penetrate a keep-out disk (h < 0)?
        self.cbf_collision_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- adaptive (online self-tuning) margin state ---
        # We estimate, each step, the rigid-model *velocity disturbance* d = ||realized - predicted||
        # of the worst body (how wrong the rigid-constellation prediction was last step -> tracking lag
        # + grasp deformation), smooth it, and set the keep-out margin to base + gain*d/alpha + vel*speed.
        self._prev_body_pos = torch.zeros(self.num_envs, n_bodies, 2, device=self.device)
        self._prev_box_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_box_yaw = torch.zeros(self.num_envs, device=self.device)
        self._prev_u = torch.zeros(self.num_envs, 3, device=self.device)  # applied body-frame twist
        self._prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cbf_disturbance = torch.zeros(self.num_envs, device=self.device)  # smoothed d (m/s)
        self.cbf_margin_effective = torch.full((self.num_envs,), float(cfg.cbf_margin), device=self.device)

        # --- constellation-morph (Idea 2a) state: tuck the grasp form in near tight obstacles ---
        # The morph scales the box-frame contact/operation offsets toward the box center, so the
        # constellation footprint shrinks; the policy observes box_pos_contact_rel via these frames, so
        # it re-grasps the moved setpoint. References are captured lazily (frame buffers are not ready
        # at command __init__) and re-captured on reset.
        self.cbf_morph_scale = torch.ones(self.num_envs, device=self.device)
        # --- v3 "regrasp" state: speed-gated creep that lines agents up along the passage axis ---
        self.cbf_regrasp_gate = torch.ones(self.num_envs, device=self.device)  # box-speed throttle (1=full)
        # --- v4 "regrasp_fsm" state: stop -> release -> reposition -> re-acquire -> resume maneuver ---
        # phases: 0 CRUISE, 1 HALT, 2 RELEASE, 3 REPOSITION, 4 REACQUIRE, 5 RESUME
        self._fsm_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fsm_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fsm_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cbf_fsm_state = torch.zeros(self.num_envs, device=self.device)  # logged phase index
        self._morph_frame_names = [f"robot_base_frame{'' if i == 0 else f'_{i}'}"
                                   for i in range(len(cfg.agent_asset_names))]
        self._morph_arm_frame_names = [f"arm_contactpad_frame{'' if i == 0 else f'_{i}'}"
                                       for i in range(len(cfg.agent_asset_names))]
        self._morph_ref = {}            # name -> default _target_frame_offset_pos clone (B*nt, 3)
        self._morph_ref_valid = False

        self.metrics["cbf_min_h"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cbf_intervention"] = torch.zeros(self.num_envs, device=self.device)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # capture the freshly-sampled command as the nominal to be filtered each step
        if hasattr(self, "nominal_command_b"):
            self.nominal_command_b[env_ids] = self.vel_command_b[env_ids]

    def _update_command(self):
        super()._update_command()
        if not hasattr(self, "nominal_command_b") or self._num_obstacles == 0:
            return

        # Current world state of the coupled bodies: box center (body 0) + robot bases.
        box_pos_xy = self.robot.data.root_pos_w[:, :2]
        _, _, box_yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        agent_xy = [a.data.root_pos_w[:, :2] for a in self._agent_assets]
        body_pos_xy = torch.stack([box_pos_xy, *agent_xy], dim=1)  # (B, Nb, 2)

        # Guidance. The *planner* part (goal-seeking) is obstacle-unaware and is shared by both modes.
        # The *avoidance* part (circulation go-around) is part of the safety layer, so it is applied
        # ONLY when the CBF is enabled -- otherwise the baseline would "avoid" via circulation and the
        # off-vs-on comparison would be meaningless. So: off = goal-seeking straight into obstacles
        # (decPLM has no obstacle concept); on = goal-seeking + circulation + CBF QP.
        u_nom4 = self._guidance_command(box_pos_xy, box_yaw, with_circulation=self.cfg.enable_cbf)
        self.vel_command_b[:, 3] = u_nom4[:, 3]  # keep the (possibly guidance-set) height command
        u_nom = u_nom4[:, :3]

        # Online self-tuning margin: estimate the rigid-model disturbance from last step and inflate
        # the keep-out so it tracks the *measured* tracking error instead of a fixed worst case.
        margin = self._adaptive_margin(box_pos_xy, box_yaw, body_pos_xy)

        # v3 topological re-grasp (speed-gated creep): when a narrow gap is approaching, slew the
        # contacts toward a fore/aft line (minimising the cross-passage envelope) and throttle the box
        # speed by the reconfiguration error + grasp disturbance. Done BEFORE the QP so the CBF filters
        # the already-throttled nominal (the creep stays provably safe -- a scaled-down safe twist).
        morph_on = getattr(self.cfg, "morph_constellation", False)
        mode = getattr(self.cfg, "morph_mode", "")
        regrasp = morph_on and mode == "regrasp"
        regrasp_fsm = morph_on and mode == "regrasp_fsm"
        if regrasp:
            self._regrasp_constellation(box_pos_xy, box_yaw)
            u_nom = u_nom.clone()
            u_nom[:, :2] = u_nom[:, :2] * self.cbf_regrasp_gate.unsqueeze(1)
        elif regrasp_fsm:
            self._regrasp_fsm(box_pos_xy, box_yaw)
            u_nom = u_nom.clone()
            u_nom[:, :2] = u_nom[:, :2] * self.cbf_regrasp_gate.unsqueeze(1)

        u_safe, diag = cbf_safety_filter(
            u_nom=u_nom,
            box_pos_xy=box_pos_xy,
            box_yaw=box_yaw,
            body_pos_xy=body_pos_xy,
            body_radius=self._body_radius,
            obstacle_xy=self._obstacle_xy,
            obstacle_radius=self._obstacle_radius,
            alpha=self.cfg.cbf_alpha,
            margin=margin,
            n_iter=self.cfg.cbf_qp_iters,
        )

        # Record diagnostics for both modes (so the baseline run reports the same safety signals).
        self.cbf_min_h = diag.min_h
        self.cbf_intervention = diag.intervention
        self.cbf_active = diag.active
        self.cbf_h_vals = diag.h_vals
        self.cbf_collision_latched |= diag.min_h < 0.0

        # Margin-free physical clearance: closest body-surface-to-obstacle-surface distance. This is
        # the *actual* collision test (< 0 == real overlap), distinct from cbf_min_h which dips below
        # zero as soon as a body enters the inflated (margin + radii) keep-out.
        dd = body_pos_xy[:, :, None, :] - self._obstacle_xy[:, None, :, :]  # (B, Nb, K, 2)
        surf = torch.norm(dd, dim=-1) - (
            self._body_radius[None, :, None] + self._obstacle_radius[None, None, :]
        )  # (B, Nb, K)
        body_clearance = surf.min(dim=2).values  # (B, Nb) per-body clearance to nearest obstacle
        self.cbf_min_clearance = body_clearance.min(dim=1).values

        # Constellation morph (Idea 2a): change the grasp form near tight obstacles. The radial tuck
        # (v1/v2) uses the post-QP physical clearance; the topological re-grasp (v3) ran before the QP.
        if morph_on and not regrasp and not regrasp_fsm:
            self._morph_constellation(body_clearance)

        # Write back the twist (height was already set above). off applies the obstacle-unaware
        # goal-seeking command directly (drives into obstacles); on applies the CBF-filtered command.
        applied = u_safe if self.cfg.enable_cbf else u_nom
        self.vel_command_b[:, :3] = applied

        # stash this step's state + applied twist so next step can measure the rigid-model disturbance
        self._prev_body_pos = body_pos_xy.detach().clone()
        self._prev_box_pos = box_pos_xy.detach().clone()
        self._prev_box_yaw = box_yaw.detach().clone()
        self._prev_u = applied.detach().clone()
        self._prev_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _adaptive_margin(self, box_pos_xy, box_yaw, body_pos_xy) -> "float | torch.Tensor":
        """Per-env keep-out margin. If ``adaptive_margin`` is off, returns the fixed ``cbf_margin``.

        Otherwise estimates the worst-body rigid-model *velocity disturbance* from the previous step
        (``d = ||realized - predicted||``, where predicted = ``A_b u`` from last step's applied twist
        and pose), smooths it (EMA), and returns
        ``clamp(margin_base + margin_gain*d/alpha + margin_vel_gain*speed, margin_base, margin_max)``.
        This grows the keep-out exactly when an agent swings wide / starts detaching (the rigid model
        is most wrong) and shrinks it in easy open stretches."""
        cfg = self.cfg
        if not getattr(cfg, "adaptive_margin", False):
            return cfg.cbf_margin

        dt = self._env.step_dt
        # predicted body world velocity from last step's pose + applied twist: vp = R(yaw) u_xy + wz*perp
        R_prev = rotation_2d(self._prev_box_yaw)  # (B,2,2)
        v_lin = torch.bmm(R_prev, self._prev_u[:, :2, None]).squeeze(-1)  # (B,2)
        lever = self._prev_body_pos - self._prev_box_pos.unsqueeze(1)  # (B,Nb,2)
        perp = torch.stack((-lever[..., 1], lever[..., 0]), dim=-1)  # (B,Nb,2)
        v_pred = v_lin.unsqueeze(1) + self._prev_u[:, 2, None, None] * perp  # (B,Nb,2)
        v_real = (body_pos_xy - self._prev_body_pos) / dt  # (B,Nb,2)
        d_body = torch.norm(v_real - v_pred, dim=-1)  # (B,Nb) per-body velocity disturbance
        d_now = d_body.max(dim=-1).values  # (B,) worst body

        # ignore the first step and reset steps (a teleport makes the finite difference meaningless)
        box_jump = torch.norm(box_pos_xy - self._prev_box_pos, dim=-1)
        valid = self._prev_valid & (box_jump < 0.3)
        d_now = torch.where(valid, d_now, torch.zeros_like(d_now))

        # Asymmetric EMA: fast attack (grow the buffer immediately when the disturbance spikes) +
        # slow release (decay gently). This avoids the lag where a symmetric EMA grows the margin a
        # step too late and the realized clearance grazes zero.
        rising = d_now > self.cbf_disturbance
        a = torch.where(rising, torch.full_like(d_now, cfg.margin_ema_attack),
                        torch.full_like(d_now, cfg.margin_ema))
        new_d = a * self.cbf_disturbance + (1.0 - a) * d_now
        self.cbf_disturbance = torch.where(valid, new_d, self.cbf_disturbance)

        box_speed = torch.norm((box_pos_xy - self._prev_box_pos) / dt, dim=-1) * valid
        margin = (cfg.margin_base
                  + cfg.margin_gain * self.cbf_disturbance / max(cfg.cbf_alpha, 1e-3)
                  + cfg.margin_vel_gain * box_speed)
        margin = margin.clamp(cfg.margin_base, cfg.margin_max)
        self.cbf_margin_effective = margin
        return margin

    def _morph_constellation(self, body_clearance: torch.Tensor):
        """Change the grasp form near tight obstacles (Idea 2a). Two modes:

        - ``"tuck"`` (v1): ALL agents tuck toward the box center by the same scale ``s ∈
          [morph_min_scale, 1]`` from the *worst* body clearance -> symmetric shrink of the footprint.
        - ``"directional"`` (v2): EACH agent tucks by its OWN clearance, so only the agent(s) near an
          obstacle pull in (away from it) while the rest keep their form -> the constellation deforms
          asymmetrically toward the safe side. More geometry-aware, in-distribution (moves where each
          agent grasps).

        Scales the box-frame contact/operation offsets' XY (keeps height); the policy observes
        ``box_pos_contact_rel`` through these frames, so it re-grasps the moved setpoint."""
        cfg = self.cfg
        lo, hi = cfg.morph_clearance_lo, cfg.morph_clearance_hi
        na = len(self._morph_frame_names)

        def scale_from(clear):  # (B,) clearance -> (B,) scale in [morph_min_scale, 1]
            frac = ((clear - lo) / max(hi - lo, 1e-3)).clamp(0.0, 1.0)
            return cfg.morph_min_scale + (1.0 - cfg.morph_min_scale) * frac

        if cfg.morph_mode == "directional":
            agent_scale = scale_from(body_clearance[:, 1:1 + na])  # (B, na) per-agent (body 0 = box)
        else:  # "tuck"
            g = scale_from(body_clearance.min(dim=1).values)  # (B,) from worst body
            agent_scale = g.unsqueeze(1).expand(-1, na)
        self.cbf_morph_scale = agent_scale.mean(dim=1)  # diagnostic (mean tuck)

        if not hasattr(self, "_morph_frames"):  # cache frame objects once (avoid per-step scene lookups)
            self._morph_frames = {}
            for nm in self._morph_frame_names + self._morph_arm_frame_names:
                try:
                    self._morph_frames[nm] = self._env.scene[nm]
                except KeyError:
                    pass
        # apply per-agent scale to that agent's base + arm contact frames
        for agent_i in range(na):
            s = agent_scale[:, agent_i]  # (B,)
            for name in (self._morph_frame_names[agent_i], self._morph_arm_frame_names[agent_i]):
                frame = self._morph_frames.get(name)
                buf = None if frame is None else getattr(frame, "_target_frame_offset_pos", None)
                if buf is None:
                    continue
                nt = len(frame._target_frame_names)  # buffer is env-major (B*nt, 3)
                if not self._morph_ref_valid or name not in self._morph_ref:
                    self._morph_ref[name] = buf.detach().clone()
                ref = self._morph_ref[name]
                buf[:, :2] = s.repeat_interleave(nt)[:, None] * ref[:, :2]  # tuck XY; keep Z
        self._morph_ref_valid = True

    def _regrasp_constellation(self, box_pos_xy: torch.Tensor, box_yaw: torch.Tensor):
        """v3 topological re-grasp with speed-gated creep.

        v1/v2 only *scale* the contacts radially -- they preserve the grasp topology (every agent stays
        on its face), so the cross-passage envelope (the agents flanking the box laterally) stays wide
        and the train jams in a narrow gap. v3 instead **re-assigns the contacts to a fore/aft line**
        along the passage axis: it keeps each agent's along-axis component, compresses its cross-axis
        (lateral) component toward the centerline, and staggers agents along the axis (ranked by their
        current lateral position, so the order is monotone and no two agents swap sides / collide). The
        result minimises the cross-passage width to ~the box width, which is the only thing that lets the
        constellation thread a gap narrower than the original footprint.

        It is **not** a snap: each step the written offset *creeps* toward the (engagement-blended) target
        at a bounded rate, and the box forward speed is **throttled** by the reconfiguration error plus
        the grasp disturbance ``d̄`` -- so the box crawls/pauses while the agents migrate and re-grip, and
        resumes once the new grasp settles. The "stop to reconfigure" behaviour therefore emerges from the
        gate rather than a hard state machine, and contact is never explicitly released."""
        cfg = self.cfg
        B = self.num_envs
        na = len(self._morph_frame_names)

        # passage axis in the BOX frame (toward the goal); e_perp is the cross-passage (lateral) axis.
        goal_world = self._env.scene.env_origins[:, :2] + self._goal_xy  # (B,2)
        to_goal = goal_world - box_pos_xy
        to_goal = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        R = rotation_2d(box_yaw)  # body->world
        e_pass = torch.bmm(R.transpose(1, 2), to_goal[:, :, None]).squeeze(-1)  # world->box (B,2)
        e_perp = torch.stack((-e_pass[:, 1], e_pass[:, 0]), dim=-1)  # (B,2)

        # engagement: ramp 0->1 as the box closes on the nearest obstacle (anticipatory).
        d = box_pos_xy[:, None, :] - self._obstacle_xy  # (B,K,2)
        gap_dist = (d.norm(dim=-1) - self._obstacle_radius[None, :]).min(dim=1).values  # (B,)
        span = max(cfg.regrasp_engage_dist - cfg.regrasp_full_dist, 1e-3)
        eng = ((cfg.regrasp_engage_dist - gap_dist) / span).clamp(0.0, 1.0)  # (B,)

        if not hasattr(self, "_morph_frames"):  # cache frame objects once
            self._morph_frames = {}
            for nm in self._morph_frame_names + self._morph_arm_frame_names:
                try:
                    self._morph_frames[nm] = self._env.scene[nm]
                except KeyError:
                    pass
        # capture default offsets once per episode (needed before we can build targets)
        for name in self._morph_frame_names + self._morph_arm_frame_names:
            frame = self._morph_frames.get(name)
            buf = None if frame is None else getattr(frame, "_target_frame_offset_pos", None)
            if buf is not None and (not self._morph_ref_valid or name not in self._morph_ref):
                self._morph_ref[name] = buf.detach().clone()
        self._morph_ref_valid = True

        # each agent's default lateral (cross) position, from its base frame -> rank -> along-axis station
        cross_all = []
        for name in self._morph_frame_names:
            ref = self._morph_ref.get(name)
            if ref is None:
                cross_all.append(torch.zeros(B, device=self.device)); continue
            ref_xy = ref.view(B, -1, 3)[..., :2]  # (B,nt,2)
            cross_all.append((ref_xy * e_perp[:, None, :]).sum(-1).mean(1))  # (B,)
        cross_all = torch.stack(cross_all, dim=1)  # (B,na)
        rank = cross_all.argsort(dim=1).argsort(dim=1).float()  # (B,na) in [0,na-1]
        station = (rank - (na - 1) / 2.0) * cfg.regrasp_spacing  # (B,na) fore/aft stations

        recon = torch.zeros(B, device=self.device)
        n_pts = 0
        for agent_i in range(na):
            st = station[:, agent_i]  # (B,)
            for name in (self._morph_frame_names[agent_i], self._morph_arm_frame_names[agent_i]):
                frame = self._morph_frames.get(name)
                buf = None if frame is None else getattr(frame, "_target_frame_offset_pos", None)
                if buf is None:
                    continue
                nt = len(frame._target_frame_names)
                ref = self._morph_ref[name].view(B, nt, 3)
                ref_xy = ref[..., :2]
                along = (ref_xy * e_pass[:, None, :]).sum(-1)  # (B,nt)
                cross = (ref_xy * e_perp[:, None, :]).sum(-1)  # (B,nt)
                t_along = along * cfg.regrasp_along_scale + st[:, None]  # spread fore/aft
                t_cross = cross * cfg.regrasp_cross_scale               # pull toward centerline
                target_xy = t_along[..., None] * e_pass[:, None, :] + t_cross[..., None] * e_perp[:, None, :]
                desired_xy = ref_xy + eng[:, None, None] * (target_xy - ref_xy)  # blend by engagement
                cur = buf.view(B, nt, 3)
                step = (desired_xy - cur[..., :2]).clamp(-cfg.regrasp_slew_rate, cfg.regrasp_slew_rate)
                new_xy = cur[..., :2] + step  # rate-limited creep
                cur[..., :2] = new_xy
                recon = recon + (new_xy - desired_xy).norm(dim=-1).mean(1)
                n_pts += 1

        recon = recon / max(n_pts, 1)  # mean residual offset (m) from the desired form
        recon_norm = recon / max(cfg.regrasp_spacing, 1e-3)
        gate = 1.0 - cfg.regrasp_gate_recon * recon_norm - cfg.regrasp_gate_dist * self.cbf_disturbance
        self.cbf_regrasp_gate = gate.clamp(cfg.regrasp_gate_min, 1.0)
        # diagnostic in the same "form scale" units as the tuck (1 = open form, -> cross_scale lined up)
        self.cbf_morph_scale = 1.0 - eng * (1.0 - cfg.regrasp_cross_scale)

    def _fore_aft_targets(self, box_pos_xy, box_yaw):
        """Per-frame fore/aft 'line' contact targets (box frame), shared by v3 and v4. Returns
        ``(e_pass, e_perp, station)`` plus a closure-free dict ``targets[name] -> (B,nt,2)`` of the
        full-engagement target XY (cross compressed, agents staggered along the passage axis)."""
        cfg = self.cfg
        B = self.num_envs
        na = len(self._morph_frame_names)
        goal_world = self._env.scene.env_origins[:, :2] + self._goal_xy
        to_goal = goal_world - box_pos_xy
        to_goal = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        R = rotation_2d(box_yaw)
        e_pass = torch.bmm(R.transpose(1, 2), to_goal[:, :, None]).squeeze(-1)
        e_perp = torch.stack((-e_pass[:, 1], e_pass[:, 0]), dim=-1)
        if not hasattr(self, "_morph_frames"):
            self._morph_frames = {}
            for nm in self._morph_frame_names + self._morph_arm_frame_names:
                try:
                    self._morph_frames[nm] = self._env.scene[nm]
                except KeyError:
                    pass
        for name in self._morph_frame_names + self._morph_arm_frame_names:
            frame = self._morph_frames.get(name)
            buf = None if frame is None else getattr(frame, "_target_frame_offset_pos", None)
            if buf is not None and (not self._morph_ref_valid or name not in self._morph_ref):
                self._morph_ref[name] = buf.detach().clone()
        self._morph_ref_valid = True
        cross_all = []
        for name in self._morph_frame_names:
            ref = self._morph_ref.get(name)
            if ref is None:
                cross_all.append(torch.zeros(B, device=self.device)); continue
            ref_xy = ref.view(B, -1, 3)[..., :2]
            cross_all.append((ref_xy * e_perp[:, None, :]).sum(-1).mean(1))
        cross_all = torch.stack(cross_all, dim=1)
        rank = cross_all.argsort(dim=1).argsort(dim=1).float()
        station = (rank - (na - 1) / 2.0) * cfg.regrasp_spacing  # (B,na)
        targets = {}
        for agent_i in range(na):
            st = station[:, agent_i]
            for name in (self._morph_frame_names[agent_i], self._morph_arm_frame_names[agent_i]):
                frame = self._morph_frames.get(name)
                if frame is None or name not in self._morph_ref:
                    continue
                nt = len(frame._target_frame_names)
                ref = self._morph_ref[name].view(B, nt, 3)
                ref_xy = ref[..., :2]
                along = (ref_xy * e_pass[:, None, :]).sum(-1)
                cross = (ref_xy * e_perp[:, None, :]).sum(-1)
                t_along = along * cfg.regrasp_along_scale + st[:, None]
                t_cross = cross * cfg.regrasp_cross_scale
                targets[name] = t_along[..., None] * e_pass[:, None, :] + t_cross[..., None] * e_perp[:, None, :]
        return targets

    def _regrasp_fsm(self, box_pos_xy, box_yaw):
        """v4 stop-disengage-reconfigure-continue maneuver. v3 showed the frozen policy will not slide a
        *held* grasp laterally across the box. v4 instead breaks the grasp first: it halts the box, lifts
        the arm contacts off the box (RELEASE), walks the contact setpoints to the fore/aft line while
        released (REPOSITION), lowers them back to re-acquire at the new stations (REACQUIRE), then drives
        through (RESUME). Phases are timed; the box is held (speed gate -> 0) for the whole maneuver and
        only resumes once re-acquired. Done once per gap (latched until the box clears the obstacle)."""
        cfg = self.cfg
        B = self.num_envs
        S, T = self._fsm_state, self._fsm_timer

        d = box_pos_xy[:, None, :] - self._obstacle_xy
        gap = (d.norm(dim=-1) - self._obstacle_radius[None, :]).min(dim=1).values  # (B,)
        ep_steps = self._env.episode_length_buf  # (B,) steps since reset

        targets = self._fore_aft_targets(box_pos_xy, box_yaw)  # ensures refs captured

        # --- transitions. Fire ONCE on entry to the obstacle field, but only after the box has lifted
        # (fsm_arm_after_steps), then HALT->RELEASE->REPOSITION->REACQUIRE->RESUME. RESUME (5) is terminal:
        # the narrow fore/aft form is HELD and the box drives through the whole corridor (both obstacles
        # sit in one narrow passage), so we do not restore mid-course. ---
        enter = (S == 0) & (gap < cfg.fsm_trigger_dist) & (ep_steps > cfg.fsm_arm_after_steps)
        S[enter] = 1; T[enter] = 0
        T = T + 1
        for st_from, st_to, dur in ((1, 2, cfg.fsm_halt_steps), (2, 3, cfg.fsm_release_steps),
                                    (3, 4, cfg.fsm_repos_steps), (4, 5, cfg.fsm_reacq_steps)):
            adv = (S == st_from) & (T >= dur)
            S[adv] = st_to; T[adv] = 0
        self._fsm_timer = T

        # --- box speed gate: hold (0) during HALT..REACQUIRE, full (1) during CRUISE/RESUME ---
        moving = (S == 0) | (S == 5)
        self.cbf_regrasp_gate = torch.where(moving, torch.ones_like(gap),
                                            torch.full_like(gap, cfg.fsm_hold_gate))

        # --- contact offsets: lift Z during RELEASE+REPOSITION; slew XY to target during REPOSITION on ---
        at_target = (S == 3) | (S == 4) | (S == 5)           # XY at fore/aft line
        lifted = (S == 2) | (S == 3)                         # arm retracted off the box
        for name, frame in self._morph_frames.items():
            buf = getattr(frame, "_target_frame_offset_pos", None)
            if buf is None or name not in self._morph_ref:
                continue
            nt = len(frame._target_frame_names)
            ref = self._morph_ref[name].view(B, nt, 3)
            cur = buf.view(B, nt, 3)
            tgt_xy = targets.get(name, ref[..., :2])
            want_xy = torch.where(at_target[:, None, None], tgt_xy, ref[..., :2])
            step_xy = (want_xy - cur[..., :2]).clamp(-cfg.regrasp_slew_rate, cfg.regrasp_slew_rate)
            cur[..., :2] = cur[..., :2] + step_xy
            want_z = ref[..., 2] + torch.where(lifted, torch.full_like(gap, cfg.fsm_release_lift),
                                               torch.zeros_like(gap))[:, None]
            step_z = (want_z - cur[..., 2]).clamp(-cfg.regrasp_slew_rate, cfg.regrasp_slew_rate)
            cur[..., 2] = cur[..., 2] + step_z

        self.cbf_fsm_state = S.float()
        self.cbf_morph_scale = 1.0 - (S > 0).float() * (1.0 - cfg.regrasp_cross_scale)

    def _guidance_command(
        self, box_pos_xy: torch.Tensor, box_yaw: torch.Tensor, with_circulation: bool = True
    ) -> torch.Tensor:
        """The (pre-CBF) nominal box twist ``[vx, vy, wz, height]`` in the box body frame.

        Without goal-seeking this is just the resampled categorical command (constant velocity). With
        ``goal_seeking`` it is a P-controller toward ``goal_xy`` (so the box returns to course after a
        deflection and reaches the next obstacle). The tangential go-around term is added only when
        ``with_circulation`` (it is part of the safety layer, applied with the CBF, not the baseline).
        """
        cfg = self.cfg
        R = rotation_2d(box_yaw)  # (B, 2, 2), maps body-frame planar vectors to world

        if cfg.goal_seeking:
            goal_world = self._env.scene.env_origins[:, :2] + self._goal_xy  # (B, 2)
            v_world = cfg.goal_kp * (goal_world - box_pos_xy)  # P-control toward the goal
            speed = torch.norm(v_world, dim=-1, keepdim=True).clamp_min(1e-6)
            v_world = v_world * (cfg.goal_speed / speed).clamp(max=1.0)  # cap at goal_speed
            yaw_rate = torch.zeros(self.num_envs, device=self.device)
            height = self.nominal_command_b[:, 3]
        else:
            u = self.nominal_command_b
            v_world = torch.bmm(R, u[:, :2, None]).squeeze(-1)  # body twist -> world velocity
            yaw_rate = u[:, 2]
            height = u[:, 3]

        if with_circulation and cfg.circulation_gain > 0.0:
            v_world = v_world + self._circulation(box_pos_xy, v_world)

        v_body = torch.bmm(R.transpose(1, 2), v_world[:, :, None]).squeeze(-1)  # world -> body
        u_nom = torch.zeros(self.num_envs, 4, device=self.device)
        u_nom[:, :2] = v_body
        u_nom[:, 2] = yaw_rate
        u_nom[:, 3] = height
        return u_nom

    def _circulation(self, box_pos_xy: torch.Tensor, v_world: torch.Tensor) -> torch.Tensor:
        """World-frame tangential velocity that steers the box *around* each nearby obstacle.

        For obstacle ``i`` the tangent direction is ``perp(box - o_i)``, with the sign chosen so the
        detour continues in the current heading; the magnitude decays with clearance so it only acts
        near the keep-out boundary. This breaks the head-on CBF deadlock (where the QP would otherwise
        zero the forward velocity and leave the box stalled at the boundary)."""
        cfg = self.cfg
        d = box_pos_xy[:, None, :] - self._obstacle_xy  # (B, K, 2) outward radial
        dist = torch.norm(d, dim=-1).clamp_min(1e-6)  # (B, K)
        n = d / dist[..., None]
        t = torch.stack([-n[..., 1], n[..., 0]], dim=-1)  # (B, K, 2) tangent (perp to radial)
        align = (t * v_world[:, None, :]).sum(-1)  # (B, K) project heading onto each tangent
        s = torch.where(align >= 0.0, 1.0, -1.0)  # detour toward the forward side
        r_eff = self._body_radius[0] + self._obstacle_radius + cfg.cbf_margin  # (K,)
        clearance = (dist - r_eff[None, :]).clamp_min(0.0)  # (B, K)
        act = torch.exp(-clearance / cfg.circulation_dist)  # ~1 at the boundary, decays outward
        circ = (cfg.circulation_gain * act * s)[..., None] * t  # (B, K, 2)
        speed = torch.norm(v_world, dim=-1, keepdim=True)  # scale tangential push by nominal speed
        return circ.sum(dim=1) * speed

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["cbf_min_h"] = self.cbf_min_h
        self.metrics["cbf_min_clearance"] = self.cbf_min_clearance
        self.metrics["cbf_intervention"] = self.cbf_intervention
        self.metrics["cbf_margin_effective"] = self.cbf_margin_effective
        self.metrics["cbf_disturbance"] = self.cbf_disturbance
        self.metrics["cbf_morph_scale"] = self.cbf_morph_scale
        self.metrics["cbf_regrasp_gate"] = self.cbf_regrasp_gate
        self.metrics["cbf_fsm_state"] = self.cbf_fsm_state

    def reset(self, env_ids: Sequence[int] | None = None):
        idx = slice(None) if env_ids is None else env_ids
        self.cbf_collision_latched[idx] = False
        # clear adaptive-margin history so a new episode does not carry over the disturbance estimate
        self._prev_valid[idx] = False
        self.cbf_disturbance[idx] = 0.0
        self.cbf_margin_effective[idx] = getattr(self.cfg, "margin_base", self.cfg.cbf_margin)
        # restore un-morphed contact/operation offsets and re-capture next episode (num_envs=1 play)
        if getattr(self.cfg, "morph_constellation", False) and self._morph_ref_valid:
            for name, ref in self._morph_ref.items():
                buf = getattr(self._env.scene[name], "_target_frame_offset_pos", None)
                if buf is not None:
                    buf[:] = ref
            self._morph_ref_valid = False
            self.cbf_morph_scale[idx] = 1.0
            self.cbf_regrasp_gate[idx] = 1.0
        # FSM state is reset unconditionally (independent of whether offsets were captured)
        self._fsm_state[idx] = 0
        self._fsm_timer[idx] = 0
        self._fsm_done[idx] = False
        self.cbf_fsm_state[idx] = 0.0
        # pass the original env_ids through so the base handles the None case itself
        return super().reset(env_ids)


@configclass
class CBFSafeVelocityHeightCommandCfg(CategoricalVelocityHeightCommandCfg):
    """Configuration for the coupled-CBF safety-filtered velocity+height command generator."""

    class_type: type = CBFSafeVelocityHeightCommand

    enable_cbf: bool = True
    """If False, the nominal command passes through unfiltered (baseline open-space policy), but the
    CBF diagnostics (``cbf_min_h`` etc.) are still computed and logged for comparison."""

    obstacles: list[tuple[float, float, float]] = field(default_factory=list)
    """Obstacles as ``(x, y, radius)`` in **env-local** coordinates (shifted per-env by the env
    origin). An empty list disables the filter entirely (pure pass-through)."""

    box_radius: float = 0.33
    """Default effective radius of the box center (used when ``body_radii`` is None)."""

    agent_radius: float = 0.35
    """Default effective radius of each robot base (used when ``body_radii`` is None)."""

    body_radii: tuple[float, ...] | None = None
    """Optional explicit per-body avoidance radii, ordered ``(box, agent_0, agent_1, ...)`` -- must
    have ``1 + len(agent_asset_names)`` entries. If None, built as ``[box_radius] + [agent_radius]*N``
    so the filter scales to any number of agents. Each keep-out disk is inflated by the sum of the
    body radius and the obstacle radius."""

    agent_asset_names: tuple[str, ...] = ("robot", "robot_1")
    """Scene asset names of the robot bases whose trajectories must also stay safe (the coupling).
    For the N-robot task this is ``("robot", "robot_1", ..., "robot_{N-1}")``."""

    cbf_alpha: float = 2.0
    """Class-:math:`\\mathcal K` gain. Larger => the filter tolerates approaching closer/faster before
    intervening (sharper, later correction); smaller => earlier, gentler deflection."""

    cbf_margin: float = 0.05
    """Fixed extra safety buffer (m) added to every radius sum. Used when ``adaptive_margin`` is
    False (and as the initial value otherwise)."""

    cbf_qp_iters: int = 200
    """Hildreth (over-relaxed projected Gauss-Seidel) dual sweeps per step."""

    # --- adaptive (online self-tuning) margin ---
    adaptive_margin: bool = False
    """If True, the keep-out margin is tuned online from the measured rigid-model disturbance instead
    of being the fixed ``cbf_margin``: ``m = clamp(margin_base + margin_gain*d/alpha +
    margin_vel_gain*speed, margin_base, margin_max)``, where ``d`` is the smoothed worst-body velocity
    mismatch (tracking lag + grasp deformation). Grows the buffer exactly where the model fails."""

    margin_base: float = 0.05
    """Floor margin (m) when ``adaptive_margin`` (open-space, well-tracked regions)."""

    margin_gain: float = 0.5
    """Gain (s) on the disturbance term ``d/alpha``; larger => more conservative when tracking is poor."""

    margin_vel_gain: float = 0.05
    """Gain (s) on box speed (a braking-distance buffer: faster => wider keep-out)."""

    margin_ema: float = 0.9
    """EMA factor on *release* (disturbance falling): closer to 1 => slower decay of the buffer."""

    margin_ema_attack: float = 0.3
    """EMA factor on *attack* (disturbance rising): small => the buffer grows fast on a spike (safety)."""

    margin_max: float = 0.45
    """Cap on the adaptive margin (m) to keep corridors passable."""

    # --- constellation morph (Idea 2a prototype): change grasp "form" near tight obstacles ---
    morph_constellation: bool = False
    """If True, tuck the grasp form (scale the box-frame contact/operation offsets toward the box
    center) as obstacles get close, so the constellation footprint shrinks. The policy observes
    ``box_pos_contact_rel`` through these frames, so it is asked to re-grasp the tighter form."""

    morph_mode: str = "tuck"
    """``"tuck"`` (v1): all agents tuck by the worst clearance (symmetric). ``"directional"`` (v2):
    each agent tucks by its own clearance, so only the threatened agent(s) pull in (asymmetric).
    ``"regrasp"`` (v3): topological re-grasp -- line the agents up fore/aft along the passage axis
    (compress the cross-passage envelope) with a speed-gated creep (see the ``regrasp_*`` knobs)."""

    # --- v3 topological re-grasp (speed-gated creep) ---
    regrasp_engage_dist: float = 2.8
    """Box-to-nearest-obstacle distance (m) at which the re-grasp starts engaging. Set large so the
    reconfiguration happens in OPEN SPACE, well before the box reaches the CBF keep-out and deadlocks."""

    regrasp_full_dist: float = 1.5
    """Distance (m) at/below which the fore/aft line is fully commanded (engagement = 1) -- before the
    pinch, so the agents have already lined up by the time the box arrives at the gap."""

    regrasp_cross_scale: float = 0.30
    """Cross-passage (lateral) compression of each contact at full engagement (0.30 = pull lateral
    offset in to 30%); this is what shrinks the envelope so the train fits the gap."""

    regrasp_along_scale: float = 1.0
    """Scale on each contact's along-passage component (1.0 keeps fore/aft spread before staggering)."""

    regrasp_spacing: float = 0.30
    """Along-axis station spacing (m) between ranked agents, so they line up fore/aft without overlap."""

    regrasp_slew_rate: float = 0.02
    """Max change (m) of a contact offset per step -- the creep rate (no snapping)."""

    regrasp_gate_recon: float = 1.2
    """Speed-throttle gain on the (normalised) reconfiguration residual: box crawls while migrating."""

    regrasp_gate_dist: float = 0.3
    """Speed-throttle gain on the grasp disturbance ``d̄``: box slows while the grasp is deforming."""

    regrasp_gate_min: float = 0.2
    """Floor on the box-speed throttle (never fully stops, so it always creeps through)."""

    # --- v4 re-grasp FSM (stop-disengage-reconfigure-continue) ---
    fsm_trigger_dist: float = 1.7
    """Box-to-nearest-obstacle distance (m) that triggers the maneuver on entry to the obstacle field."""

    fsm_arm_after_steps: int = 120
    """Only arm the trigger after this many steps since reset, so the box has finished reach->touch->lift
    before we release the grasp (releasing pre-lift would be nonsense)."""

    fsm_halt_steps: int = 25
    """Steps to hold the box still before releasing (let it come to a stop)."""

    fsm_release_steps: int = 40
    """Steps to lift the arm contacts off the box (disengage the grasp)."""

    fsm_repos_steps: int = 90
    """Steps to walk the (released) contact setpoints to the fore/aft line."""

    fsm_reacq_steps: int = 60
    """Steps to lower the contacts back down and re-acquire the grasp at the new stations."""

    fsm_release_lift: float = 0.25
    """How far (m) to raise the contact offset Z to retract the arm off the box during RELEASE/REPOSITION."""

    fsm_hold_gate: float = 0.0
    """Box-speed throttle during the maneuver (0 = fully halt while reconfiguring)."""

    morph_min_scale: float = 0.7
    """Smallest offset scale at ``morph_clearance_lo`` (1.0 = unchanged form; 0.7 = 30% tuck)."""

    morph_clearance_lo: float = 0.15
    """Physical clearance (m) at/below which the morph is fully applied (``morph_min_scale``)."""

    morph_clearance_hi: float = 0.8
    """Physical clearance (m) at/above which the form is unchanged (scale 1.0)."""

    # --- guidance (so a whole obstacle course is traversed, not just the first obstacle) ---
    goal_seeking: bool = False
    """If True, the nominal command is a P-controller driving the box toward ``goal_xy`` (instead of
    a constant velocity). This makes the box return to course after a deflection and reach the next
    obstacle, so every obstacle along the way influences the trajectory. Overrides the CLI velocity."""

    goal_xy: tuple[float, float] = (8.0, 0.0)
    """Goal point in **env-local** coordinates (shifted per-env by the env origin)."""

    goal_speed: float = 0.4
    """Cap on the goal-seeking nominal speed (m/s)."""

    goal_kp: float = 1.0
    """Proportional gain of the goal-seeking controller (1/s)."""

    circulation_gain: float = 0.0
    """Strength of the tangential "go-around" term near obstacles (0 disables it). ~1.0 gives a
    tangential push comparable to the nominal speed at the keep-out boundary, so the box rounds a
    head-on obstacle instead of stalling against it (CBF deadlock)."""

    circulation_dist: float = 0.6
    """Clearance decay length (m) of the circulation term; it is strongest at the keep-out boundary
    and falls off over this distance."""
