"""Decentralized *consensus* pinch-lift-move task.

This task reuses the full reach-touch-lift-and-move (finetuning) machinery but changes two things,
so that the common goal (box trajectory) is *implicit* and divergent per agent:

1. The box command becomes a :class:`ConsensusVelocityHeightCommand`: a single shared *mean* command
   plus an independently-resampled divergence. Each actor observes a different belief
   (``mean + sign * divergence``); the reward still tracks the *mean*, so the optimum is a physical
   compromise (consensus-to-mean). Set ``divergence_range = (0, 0, 0, 0)`` to recover the shared-goal
   task.

2. Each agent gains two partner-sensing channels -- the partner's contact force on the box and the
   partner's base pose in the agent's own frame -- so it can feel/see and adapt to its partner without
   any explicit communication of policies or goals.

The centralized critic keeps the true mean command (CTDE): actors are decentralized and only see their
own belief + partner sensing, while the critic may see ground truth.
"""

from __future__ import annotations

import os

from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

import drail_extensions.research_bikram.mdp as mdp

from ..go2_d1_marl_box_reach_touch_lift_and_move_finetuning import (
    go2_d1_marl_box_reach_touch_lift_and_move_finetuning_env_cfg as ft_env_cfg,
)

# Reuse the box command ranges/limits from the finetuning task.
BOX_VELOCITY_COMMAND_LIMIT = ft_env_cfg.BOX_VELOCITY_COMMAND_LIMIT
BOX_VELOCITY_DEAD_ZONE = ft_env_cfg.BOX_VELOCITY_DEAD_ZONE
HEIGHT_COMMAND_RANGE = ft_env_cfg.HEIGHT_COMMAND_RANGE

# Per-agent divergence of the *belief* about the box command on
# (lin_vel_x [m/s], lin_vel_y [m/s], ang_vel_z [rad/s], height [m]). Kept small so from-scratch
# training can find the consensus basin; raise it (or curriculum it) to study disturbance tolerance.
BELIEF_DIVERGENCE_RANGE = (0.15, 0.15, 0.1, 0.05)
BELIEF_DIVERGENCE_RESAMPLING_TIME_RANGE = (2.0, 4.0)


# ----------------------------------------------------------------------------------------------------
# Ablation hooks (read once at import time, set by the launch scripts via environment variables).
#   CONSENSUS_DIVERGENCE_SCALE : float multiplier on BELIEF_DIVERGENCE_RANGE. 0.0 => shared-goal
#                                baseline (beliefs == mean). Default 1.0.
#   CONSENSUS_SENSING          : which partner-sensing channels the agents get. One of
#                                {both, none, force, pose} or a comma list like "force,pose".
#                                Default "both". NOTE: this changes the observation dim, so a policy
#                                must be evaluated with the SAME CONSENSUS_SENSING it was trained with.
# ----------------------------------------------------------------------------------------------------
def _divergence_scale() -> float:
    return float(os.environ.get("CONSENSUS_DIVERGENCE_SCALE", "1.0"))


def _sensing_channels() -> set[str]:
    mode = os.environ.get("CONSENSUS_SENSING", "both").strip().lower()
    if mode in ("both", "all", ""):
        return {"force", "pose"}
    if mode in ("none", "off"):
        return set()
    return {m.strip() for m in mode.split(",") if m.strip() in ("force", "pose")}


SCALED_DIVERGENCE_RANGE = tuple(d * _divergence_scale() for d in BELIEF_DIVERGENCE_RANGE)


@configclass
class CommandsCfg:
    """The box command is the *team mean*; per-agent beliefs are formed in the observations."""

    box_command = mdp.ConsensusVelocityHeightCommandCfg(
        asset_name="box",
        resampling_time_range=(3.0, 10.0),
        debug_vis=False,
        divergence_range=SCALED_DIVERGENCE_RANGE,
        divergence_resampling_time_range=BELIEF_DIVERGENCE_RESAMPLING_TIME_RANGE,
        ranges={
            "all": mdp.ConsensusVelocityHeightCommandCfg.Ranges(
                lin_vel_x=(-BOX_VELOCITY_COMMAND_LIMIT[0], BOX_VELOCITY_COMMAND_LIMIT[0]),
                lin_vel_y=(-BOX_VELOCITY_COMMAND_LIMIT[1], BOX_VELOCITY_COMMAND_LIMIT[1]),
                ang_vel_z=(-BOX_VELOCITY_COMMAND_LIMIT[2], BOX_VELOCITY_COMMAND_LIMIT[2]),
                height=HEIGHT_COMMAND_RANGE,
                velocity_dead_zone=BOX_VELOCITY_DEAD_ZONE,
                probability=0.6,
            ),
            "height": mdp.ConsensusVelocityHeightCommandCfg.Ranges(
                lin_vel_x=(-0.0, 0.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-0.0, 0.0),
                height=HEIGHT_COMMAND_RANGE,
                velocity_dead_zone=BOX_VELOCITY_DEAD_ZONE,
                probability=0.4,
            ),
        },
    )


@configclass
class ObservationsCfg(ft_env_cfg.ObservationsCfg):
    """Adds divergent beliefs (actors) + partner sensing (force & pose) on top of the finetuning obs."""

    def __post_init__(self):
        super().__post_init__()

        # 1) Divergent beliefs: actors observe mean +/- divergence; critics keep the true mean (CTDE).
        self.policy.box_command.func = mdp.transformed_belief_commands
        self.policy.box_command.params["belief_sign"] = 1.0
        self.policy_1.box_command.func = mdp.transformed_belief_commands
        self.policy_1.box_command.params["belief_sign"] = -1.0
        # self.critic / self.critic_1 stay on transformed_commands_v4 (== mean).

        # 2) Partner sensing on every group (kept symmetric so actor/critic dims align). Which channels
        #    are present is controlled by CONSENSUS_SENSING (for ablations).
        #    Agent 0's partner is robot_1 / contact_sensor_contactpad_1; agent 1's is robot / contact_sensor_contactpad.
        sensing = _sensing_channels()
        for group, agent_id in [
            (self.policy, 0),
            (self.critic, 0),
            (self.policy_1, 1),
            (self.critic_1, 1),
        ]:
            self_suffix = "" if agent_id == 0 else "_1"
            partner_suffix = "_1" if agent_id == 0 else ""

            # "Feel the forces the partner applies to the object": partner contact-pad force on the box.
            if "force" in sensing:
                group.partner_contact_force = ObsTerm(
                    func=mdp.contact_forces,
                    params={
                        "sensor_cfg": SceneEntityCfg(f"contact_sensor_contactpad{partner_suffix}"),
                        "filter_prim_paths_expr": ["{ENV_REGEX_NS}/box"],
                        "combine_mode": "none",
                    },
                )

            # "Look at the partner's current pose": partner base pose in this agent's base frame.
            if "pose" in sensing:
                group.partner_base_pose = ObsTerm(
                    func=mdp.partner_pose_b,
                    params={
                        "asset_cfg": SceneEntityCfg(f"robot{self_suffix}"),
                        "partner_asset_cfg": SceneEntityCfg(f"robot{partner_suffix}"),
                    },
                )


@configclass
class Go2_D1_MARL_Box_Consensus_Lift_and_Move_Env_Cfg(
    ft_env_cfg.Go2_D1_MARL_Box_Reach_Touch_Lift_and_Move_Finetuning_Env_Cfg
):
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()


@configclass
class Go2_D1_MARL_Box_Consensus_Lift_and_Move_Env_Cfg_PLAY(Go2_D1_MARL_Box_Consensus_Lift_and_Move_Env_Cfg):
    """Eval variant: fewer envs, no observation corruption, long episodes. Keeps the consensus command
    active so divergent beliefs / partner sensing can be inspected during rollouts."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 999999.0
        for obs_grp in [self.observations.policy, self.observations.policy_1]:
            obs_grp.enable_corruption = False
