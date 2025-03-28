# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Pick-Place-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cam_ik_rel_env_cfg:FrankaPickPlaceCamEnvCfg",
    },
    disable_env_checker=True,
)
