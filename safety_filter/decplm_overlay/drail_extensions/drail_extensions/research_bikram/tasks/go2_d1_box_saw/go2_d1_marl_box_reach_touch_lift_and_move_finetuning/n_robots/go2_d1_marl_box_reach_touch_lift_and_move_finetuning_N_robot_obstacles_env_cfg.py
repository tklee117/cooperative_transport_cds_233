r"""N-robot obstacle stress-test with the coupled-CBF safety filter.

Same idea as the 2-robot obstacle env, scaled to the N-robot play task (4 agents surrounding an
0.8 x 0.8 box, one per face). The single shared ``box_command`` is wrapped by the coupled CBF QP so
the box and *all N robot bases* avoid the obstacles -- the QP now has one barrier per
(body in {box, agent_0, ..., agent_{N-1}}) x obstacle, all sharing the one box-twist control. The
pretrained decentralized policy is reused unchanged (the N-robot play already applies the 2-robot
policy to all agents).

Registered as ``Go2-D1-MARL-Box-Reach-Touch-Lift-and-Move-Finetuning-N-Robots-Obstacles-Play-v0`` and
driven by the same ``scripts/play_cbf.py`` (it reads agent count, obstacles and radii from the CBF
term, so the figures/GIF are produced automatically for N agents).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

import drail_extensions.research_bikram.mdp as mdp

from .go2_d1_marl_box_reach_touch_lift_and_move_finetuning_N_robot_env_cfg import (
    Go2_D1_MARL_Box_Reach_Touch_Lift_and_Move_Finetuning_N_Robots_Env_Cfg_PLAY,
    SceneCfg,
    _agent_suffix,
    ma_config,
)

# Obstacles in env-local coordinates: (x, y, radius) [m]. The box (0.8x0.8) is commanded along +x;
# obstacles ahead and off-centerline obstruct the leading agents/box.
OBSTACLES: list[tuple[float, float, float]] = [
    (2.4, 0.65, 0.35),
    (4.6, -0.65, 0.35),
]
GOAL_XY: tuple[float, float] = (6.0, 0.0)
OBSTACLE_CYL_HEIGHT = 1.0

# Wide elevated recorded-video camera (env-local) framing the whole N-robot course.
CAMERA_EYE: tuple[float, float, float] = (2.8, -8.5, 6.0)
CAMERA_LOOKAT: tuple[float, float, float] = (3.0, 0.0, 0.3)

_NUM_AGENTS = ma_config["shared"]["num_agents"]
_AGENT_ASSET_NAMES = tuple(f"robot{_agent_suffix(i)}" for i in range(_NUM_AGENTS))


def _obstacle_cfg(x: float, y: float, radius: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",  # renamed uniquely by the caller
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
class ObstacleSceneCfg_N(SceneCfg):
    """N-robot scene + static obstacle cylinders (one declared slot per obstacle)."""

    obstacle_0: RigidObjectCfg = None
    obstacle_1: RigidObjectCfg = None
    obstacle_2: RigidObjectCfg = None
    obstacle_3: RigidObjectCfg = None

    def __post_init__(self):
        super().__post_init__()
        assert len(OBSTACLES) <= 4, "Add more obstacle_i slots for >4 obstacles."
        for i, (x, y, r) in enumerate(OBSTACLES):
            cfg = _obstacle_cfg(x, y, r)
            cfg.prim_path = "{ENV_REGEX_NS}/obstacle_%d" % i
            setattr(self, f"obstacle_{i}", cfg)


def _cbf_box_command() -> "mdp.CBFSafeVelocityHeightCommandCfg":
    """CBF-filtered shared box command for all N agents (installed after super().__post_init__())."""
    rng = ma_config["shared"]["box_command_range"]
    return mdp.CBFSafeVelocityHeightCommandCfg(
        asset_name="box",
        resampling_time_range=(3.0, 10.0),
        debug_vis=False,
        enable_cbf=True,  # toggled at runtime by scripts/play_cbf.py (--cbf on|off)
        obstacles=OBSTACLES,
        agent_asset_names=_AGENT_ASSET_NAMES,
        box_radius=0.55,   # 0.8x0.8 box circumradius ~0.566
        agent_radius=0.35,
        # Conservative gains: the 4-agent constellation lags the command, so gentle goal-seeking +
        # mild circulation + a larger margin keep the realized trajectory safe while weaving past the
        # obstacles (aggressive guidance commands maneuvers the policy cannot track).
        cbf_alpha=1.5,
        cbf_margin=0.15,
        # Adaptive margin: the 4-agent constellation swings wider, so the disturbance term naturally
        # produces a larger margin here than for 2 agents -- auto-tuning, no per-case hand-tuning.
        adaptive_margin=True,
        margin_base=0.08,
        margin_gain=0.6,
        margin_vel_gain=0.05,
        margin_ema=0.9,
        margin_ema_attack=0.3,
        margin_max=0.6,
        cbf_qp_iters=200,
        # guidance: goal past the obstacles + a mild tangential go-around term.
        goal_seeking=True,
        goal_xy=GOAL_XY,
        goal_speed=0.3,
        goal_kp=0.8,
        circulation_gain=0.4,
        circulation_dist=0.8,
        ranges={
            # velocity comes from goal-seeking guidance; this range only supplies the height command.
            "all": mdp.CBFSafeVelocityHeightCommandCfg.Ranges(
                lin_vel_x=rng["lin_vel_x"],
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                height=rng["height"],
                velocity_dead_zone=rng["velocity_dead_zone"],
                probability=1.0,
            ),
        },
    )


@configclass
class Go2_D1_MARL_Box_Reach_Touch_Lift_and_Move_Finetuning_N_Robots_Obstacles_Env_Cfg_PLAY(
    Go2_D1_MARL_Box_Reach_Touch_Lift_and_Move_Finetuning_N_Robots_Env_Cfg_PLAY
):
    scene: ObstacleSceneCfg_N = ObstacleSceneCfg_N(num_envs=1, env_spacing=18.0)

    def __post_init__(self):
        super().__post_init__()
        # the N-robot play __post_init__ has installed a GUI-keyboard box command and set num_envs=50;
        # override for the single-env obstacle stress test and swap in the CBF-filtered command.
        self.scene.num_envs = 1
        self.scene.env_spacing = 18.0
        self.commands.box_command = _cbf_box_command()

        # Wide elevated camera so the recorded video shows the whole course (all obstacles + weave).
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = CAMERA_EYE
        self.viewer.lookat = CAMERA_LOOKAT
        self.viewer.resolution = (1920, 1080)
