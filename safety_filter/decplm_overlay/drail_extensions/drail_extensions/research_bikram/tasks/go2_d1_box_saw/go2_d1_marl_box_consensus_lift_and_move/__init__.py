import gymnasium as gym

from . import (
    go2_d1_marl_box_consensus_lift_and_move_env_cfg,
    rsl_rl_go2_d1_marl_box_consensus_lift_and_move_ppo_cfg,
)

gym.register(
    id="Go2-D1-MARL-Box-Consensus-Lift-and-Move-v0",
    entry_point="drail_extensions.core.isaaclab.envs:ManagerBasedMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            go2_d1_marl_box_consensus_lift_and_move_env_cfg.Go2_D1_MARL_Box_Consensus_Lift_and_Move_Env_Cfg
        ),
        "rsl_rl_cfg_entry_point": (
            rsl_rl_go2_d1_marl_box_consensus_lift_and_move_ppo_cfg.Go2_D1_MARL_Box_Consensus_Lift_and_Move_PPO_Runner_Cfg
        ),
    },
)

gym.register(
    id="Go2-D1-MARL-Box-Consensus-Lift-and-Move-Play-v0",
    entry_point="drail_extensions.core.isaaclab.envs:ManagerBasedMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            go2_d1_marl_box_consensus_lift_and_move_env_cfg.Go2_D1_MARL_Box_Consensus_Lift_and_Move_Env_Cfg_PLAY
        ),
        "rsl_rl_cfg_entry_point": (
            rsl_rl_go2_d1_marl_box_consensus_lift_and_move_ppo_cfg.Go2_D1_MARL_Box_Consensus_Lift_and_Move_PPO_Runner_Cfg
        ),
    },
)
