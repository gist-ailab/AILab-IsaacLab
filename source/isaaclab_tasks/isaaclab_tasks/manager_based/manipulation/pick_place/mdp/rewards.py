# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    plane_x_size: float = 0.18,  # 평면 X방향 크기
    plane_y_size: float = 0.262, # 평면 Y방향 크기
    z_threshold: float = 0.144,  # Z방향 거리 임계값
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    klt_cfg: SceneEntityCfg = SceneEntityCfg("small_klt"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    klt: RigidObject = env.scene[klt_cfg.name]

    # compute the desired position in the world frame
    des_pos_w = klt.data.root_pos_w[:, :3]

    # object position in the world frame
    obj_pos_w = object.data.root_pos_w[:, :3]
    
    # 평면 경계 계산 (평면 중심에서 각 방향으로의 반경)
    half_x = plane_x_size / 2.0
    half_y = plane_y_size / 2.0
    
    # 평면의 XY 범위 내에 물체가 있는지 확인
    x_in_bounds = (obj_pos_w[:, 0] >= (des_pos_w[:, 0] - half_x)) & (obj_pos_w[:, 0] <= (des_pos_w[:, 0] + half_x))
    y_in_bounds = (obj_pos_w[:, 1] >= (des_pos_w[:, 1] - half_y)) & (obj_pos_w[:, 1] <= (des_pos_w[:, 1] + half_y))
    xy_in_bounds = x_in_bounds & y_in_bounds
    
    # Z방향 거리 계산
    z_distance = torch.abs(obj_pos_w[:, 2] - des_pos_w[:, 2])
    
    # X 방향 거리 계산 (평면 경계까지의 최단 거리)
    x_left_out = torch.clamp(des_pos_w[:, 0] - half_x - obj_pos_w[:, 0], min=0.0)
    x_right_out = torch.clamp(obj_pos_w[:, 0] - (des_pos_w[:, 0] + half_x), min=0.0)
    x_dist = x_left_out + x_right_out
    
    # Y 방향 거리 계산 (평면 경계까지의 최단 거리)
    y_bottom_out = torch.clamp(des_pos_w[:, 1] - half_y - obj_pos_w[:, 1], min=0.0)
    y_top_out = torch.clamp(obj_pos_w[:, 1] - (des_pos_w[:, 1] + half_y), min=0.0)
    y_dist = y_bottom_out + y_top_out
    
    # XY 평면상 최단 거리
    xy_dist = torch.sqrt(x_dist**2 + y_dist**2)
    
    # 물체가 평면 XY 범위 내에 있는 경우와 그렇지 않은 경우를 구분하여 거리 계산
    # 평면 내부: Z 거리만 사용
    # 평면 외부: XY 최단 거리 + Z 거리 + z_threshold
    distance = torch.where(
        xy_in_bounds,
        z_distance,  # 범위 내에 있을 때
        xy_dist + z_distance + z_threshold  # 범위 밖에 있을 때
    )
    
    # 물체가 최소 높이 이상에 있는 경우에만 보상 제공
    lifted = obj_pos_w[:, 2] > minimal_height
    return lifted * (1 - torch.tanh(distance / std))
