# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp.observations as mdp
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, cuRoboIKActionCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

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
            clipping_range=(0.1, 3)
        ),
        # offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        # offset=CameraCfg.OffsetCfg(pos=(1.3, 0.0, 1.3), rot=(0.92388, 0.0, 0.38268, 0.0), convention="opengl"),
        offset=CameraCfg.OffsetCfg(pos=(1.3, 0.0, 1.3), rot=(0.65328, 0.2706, 0.2706, 0.65328), convention="opengl"),
    )

    def __post_init__(self):
        # Set camera in scene first
        self.scene.camera = self.camera

        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos={
            "panda_joint1": 0.297,
            "panda_joint2": 0.384,
            "panda_joint3": 0.0,
            "panda_joint4": -2.007,
            "panda_joint5": 0.0,
            "panda_joint6": 2.461,
            "panda_joint7": 1.047,
            "panda_finger_joint.*": 0.04,
        }

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            # scale=0.5,  # input device의 입력 값을 0.5배로 줄여서 보냄
            scale=1,  # input device의 입력 값을 0.5배로 줄여서 보냄
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


        # # TODO: 아래 부분에 cuRobo cfg 구현.
        # tensor_args = TensorDeviceType()

        # config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        # urdf_file = config_file["robot_cfg"]["kinematics"][
        #     "urdf_path"
        # ]  # Send global path starting with "/"
        # base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        # ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
        # robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

        # self.actions.arm_action = cuRoboIKActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_joint.*"],
        #     body_name="panda_hand",
        #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        #     scale=0.5,  # input device의 입력 값을 0.5배로 줄여서 보냄
        #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        # )

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
