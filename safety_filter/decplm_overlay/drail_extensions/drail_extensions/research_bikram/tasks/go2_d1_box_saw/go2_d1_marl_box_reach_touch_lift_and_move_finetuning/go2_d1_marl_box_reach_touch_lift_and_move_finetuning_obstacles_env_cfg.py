r"""Obstacle stress-test play environment for decPLM with a coupled-CBF safety filter.

This environment takes the trained decPLM finetuning *play* task and drops obstacles into the
otherwise-open workspace. The box high-level command (``box_command``) is swapped for a
:class:`CBFSafeVelocityHeightCommandCfg`, which wraps the nominal box-frame twist in a control-barrier
QP that keeps the box **and both robot bases** outside every obstacle (see
:mod:`...mdp.safety.cbf_qp`). The learned decentralized policy is unchanged and observes the *filtered*
command, so no retraining is needed and the observation space is identical to the open-space task --
the pretrained checkpoint loads directly.

Two questions this environment is built to answer:

1. **Does the open-space policy break with obstacles?** Set ``enable_cbf=False`` (CLI ``--cbf off``):
   the nominal command passes through and the constellation drives straight into the obstacle.
2. **Does the CBF filter fix it without destroying the grasp?** With ``enable_cbf=True`` the QP
   deflects the *shared* box command; the open question is whether agent 2 (which only senses the
   object and its own body) keeps the grasp while agent 1's path is bent around the obstacle.

Registered as ``Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-Obstacles-Play-v0`` and driven by
``scripts/play_cbf.py`` (which logs safety diagnostics and toggles the filter).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

import drail_extensions.research_bikram.mdp as mdp

from .go2_d1_marl_box_reach_touch_lift_and_move_finetuning_env_cfg import (
    BOX_VELOCITY_COMMAND_LIMIT,
    BOX_VELOCITY_DEAD_ZONE,
    HEIGHT_COMMAND_RANGE,
    Go2_D1_MARL_Box_Reach_Touch_Lift_and_Move_Finetuning_Env_Cfg_PLAY,
    SceneCfg,
)

# ---------------------------------------------------------------------------------------------------
# Obstacle layout, in **env-local** coordinates: (x, y, radius) [m]. The box starts at the env origin
# and is commanded along +x at play time, so obstacles placed ahead and slightly off the centerline
# sit in the swept path of the lead robot/box. Keep |y| < (lead reach + keep-out) so they actually
# obstruct. To change the count, also add/remove ``obstacle_i`` slots in ``ObstacleSceneCfg``.
# ---------------------------------------------------------------------------------------------------
# A slalom: obstacles alternate sides near the centerline so the goal-seeking box must weave past
# every one. Spaced within the distance the box covers in an episode (~0.15-0.2 m/s) so all are
# reachable. The goal (GOAL_XY) is just past the last obstacle.
OBSTACLES: list[tuple[float, float, float]] = [
    (1.8, 0.40, 0.30),
    (3.6, -0.40, 0.30),
]
GOAL_XY: tuple[float, float] = (4.8, 0.0)
OBSTACLE_CYL_HEIGHT = 1.0  # m, tall enough that the box/robots cannot pass over it

# Recorded-video camera (env-local frame): a wide, elevated 3/4 shot framing the whole course
# (start -> goal -> both obstacles) so the box never walks out of frame. Tweak EYE/LOOKAT to taste.
CAMERA_EYE: tuple[float, float, float] = (2.0, -7.0, 5.0)
CAMERA_LOOKAT: tuple[float, float, float] = (2.4, 0.0, 0.3)


def _obstacle_cfg(x: float, y: float, radius: float) -> RigidObjectCfg:
    """A static (kinematic) collision cylinder at env-local ``(x, y)`` so it repeats per-env."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",  # overwritten by the caller to a unique name
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=OBSTACLE_CYL_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, OBSTACLE_CYL_HEIGHT / 2.0)),
    )


@configclass
class ObstacleSceneCfg(SceneCfg):
    """decPLM finetuning scene + static obstacle cylinders (one declared slot per obstacle)."""

    obstacle_0: RigidObjectCfg = None
    obstacle_1: RigidObjectCfg = None
    obstacle_2: RigidObjectCfg = None
    obstacle_3: RigidObjectCfg = None

    def __post_init__(self):
        super().__post_init__()
        assert len(OBSTACLES) <= 4, "Add more obstacle_i slots to ObstacleSceneCfg for >4 obstacles."
        for i, (x, y, r) in enumerate(OBSTACLES):
            cfg = _obstacle_cfg(x, y, r)
            cfg.prim_path = "{ENV_REGEX_NS}/obstacle_%d" % i
            setattr(self, f"obstacle_{i}", cfg)


def _cbf_box_command() -> "mdp.CBFSafeVelocityHeightCommandCfg":
    """Build the CBF-filtered box command (a Categorical term, so it works headless -- no keyboard).

    NOTE: an ancestor play ``__post_init__`` replaces ``box_command`` with a GUI-keyboard command, so
    this must be installed *after* ``super().__post_init__()`` (see the env cfg below)."""
    return mdp.CBFSafeVelocityHeightCommandCfg(
        asset_name="box",
        resampling_time_range=(3.0, 10.0),
        debug_vis=False,
        # --- CBF safety-filter settings (enable_cbf toggled at runtime by scripts/play_cbf.py) ---
        enable_cbf=True,
        obstacles=OBSTACLES,
        agent_asset_names=("robot", "robot_1"),
        body_radii=(0.33, 0.35, 0.35),  # (box, agent-0 base, agent-1 base) effective radii
        # Gentler than a pure stress test: the box/agents lag the command, so aggressive guidance
        # (fast goal-seeking + strong circulation) commands maneuvers the policy cannot track and the
        # constellation overshoots into obstacles. These conservative gains + a larger margin keep the
        # *realized* trajectory safe while still weaving past every obstacle.
        cbf_alpha=1.5,
        # Fixed robust margin (used if adaptive_margin=False): from the sweep, m=0.16 gives ~+0.10 m
        # realized clearance for the 2-robot gentle guidance. Override with play_cbf.py --cbf_margin.
        cbf_margin=0.16,
        # Adaptive (online self-tuning) margin: grows the keep-out from the measured tracking
        # disturbance instead of a fixed value, so it auto-tunes per environment / agent count.
        adaptive_margin=True,
        margin_base=0.08,
        margin_gain=0.6,
        margin_vel_gain=0.05,
        margin_ema=0.9,
        margin_ema_attack=0.3,
        margin_max=0.45,
        cbf_qp_iters=200,
        # guidance: drive the box to a goal past the obstacles (so it returns to course after each
        # deflection and reaches every obstacle) + a mild tangential term to round head-on obstacles.
        goal_seeking=True,
        goal_xy=GOAL_XY,
        goal_speed=0.3,
        goal_kp=0.8,
        circulation_gain=0.4,
        circulation_dist=0.7,
        ranges={
            # velocity is set by goal-seeking guidance; this range only supplies the height (lift)
            # command. (lin_vel ranges are ignored when goal_seeking=True.)
            "all": mdp.CBFSafeVelocityHeightCommandCfg.Ranges(
                lin_vel_x=(-BOX_VELOCITY_COMMAND_LIMIT[0], BOX_VELOCITY_COMMAND_LIMIT[0]),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                height=HEIGHT_COMMAND_RANGE,
                velocity_dead_zone=BOX_VELOCITY_DEAD_ZONE,
                probability=1.0,
            ),
        },
    )


@configclass
class Go2_D1_MARL_Box_Reach_Touch_Lift_and_Move_Finetuning_Obstacles_Env_Cfg_PLAY(
    Go2_D1_MARL_Box_Reach_Touch_Lift_and_Move_Finetuning_Env_Cfg_PLAY
):
    # Single env by default for the obstacle stress test; large spacing so per-env obstacles never
    # interfere with neighbours when batching.
    scene: ObstacleSceneCfg = ObstacleSceneCfg(num_envs=1, env_spacing=15.0)

    def __post_init__(self):
        super().__post_init__()
        # The inherited play __post_init__ has, by now, installed a GUI-keyboard box command. Swap in
        # the CBF-filtered categorical command (drives the box headless and applies obstacle
        # avoidance). The obs box_command term has clip_range=None (see the finetuning obs cfg), so no
        # clip_range entry is needed here.
        self.commands.box_command = _cbf_box_command()

        # Wide elevated camera so the recorded video shows the whole course (all obstacles + the full
        # weave), not just the start. origin_type="env" anchors it to env 0's origin.
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = CAMERA_EYE
        self.viewer.lookat = CAMERA_LOOKAT
        self.viewer.resolution = (1920, 1080)
