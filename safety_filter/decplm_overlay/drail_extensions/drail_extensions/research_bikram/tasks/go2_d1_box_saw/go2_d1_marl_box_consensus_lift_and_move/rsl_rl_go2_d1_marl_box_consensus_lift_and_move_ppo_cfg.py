# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import random

from isaaclab.utils import configclass

from ..go2_d1_marl_box_reach_touch_lift_and_move import (
    rsl_rl_go2_d1_marl_box_reach_touch_lift_and_move_ppo_cfg,
)


@configclass
class Go2_D1_MARL_Box_Consensus_Lift_and_Move_PPO_Runner_Cfg(
    rsl_rl_go2_d1_marl_box_reach_touch_lift_and_move_ppo_cfg.Go2_D1_MARL_Box_Reach_Touch_Lift_and_Move_PPO_Runner_Cfg
):
    experiment_name = "go2_d1_marl_box_consensus_lift_and_move"
    seed = random.randint(0, 1000000)

    max_iterations = 1_000_000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "rsl_rl_demarl.algorithms.PPO"
        self.algorithm.entropy_coef = 0.01
        # The go2 symmetry data-augmentation assumes a fixed observation layout; our added belief /
        # partner-sensing terms change it, so disable symmetry augmentation here.
        self.algorithm.symmetry_cfg = None
