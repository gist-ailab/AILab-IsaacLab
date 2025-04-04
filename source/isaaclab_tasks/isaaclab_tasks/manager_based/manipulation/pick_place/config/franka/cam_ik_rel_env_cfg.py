# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp.observations as mdp
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaPickPlaceCamEnvCfg(joint_pos_env_cfg.FrankaPickPlaceEnvCfg):
    
    # # Azure Kinect configuration
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Table/cam",
    #     update_period=0.1,
    #     height=576,
    #     width=640,
    #     data_types=["rgb", "depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=0.18, focus_distance=1.2,
    #         horizontal_aperture=0.504, vertical_aperture=0.436,
    #         # clipping_range=(0.5, 3.86)  # Azure Kinect clipping range
    #         clipping_range=(0.2, 1.3)  # clipping table
    #     ),
    #     # offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.42), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    #     # offset=CameraCfg.OffsetCfg(pos=(0.0, -0.86, 1.0), rot=(0.9238, 0.38268, 0.0, 0.0), convention="ros"),
    #     # offset=CameraCfg.OffsetCfg(pos=(0.0, -0.86, 1.0), rot=(0.92388, 0.38268, 0.0, 0.0), convention="opengl"),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, -0.2, 1.3), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
    # )

    # Default camera configuration
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Table/cam",
        update_period=0.1,
        height=576,
        width=640,
        data_types=["rgb", "depth", "instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.1, 1.5)  # clipping table
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
    )

    def __post_init__(self):
        # Set camera in scene first
        self.scene.camera = self.camera

        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # 이게 True면 observation manager에서 오류 발생. 다른 차원의 observation을 concat 안하도록 해야 함.
        self.observations.policy.concatenate_terms = False

        # Set always enable cameras
        self.sim.enable_cameras = True


@configclass
class FrankaPickPlaceCamEnvCfg_PLAY(FrankaPickPlaceCamEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
