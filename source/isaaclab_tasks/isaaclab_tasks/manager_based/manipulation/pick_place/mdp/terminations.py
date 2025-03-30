# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    z_threshold: float = 0.06,  # Z방향 거리 임계값. 살짝 마진 둠
    plane_x_size: float = 0.18,  # 평면 X방향 크기
    plane_y_size: float = 0.262, # 평면 Y방향 크기
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    klt_cfg: SceneEntityCfg = SceneEntityCfg("small_klt"),
) -> torch.Tensor:
    """평면 영역 위에 물체가 위치하는지 확인하는 종료 조건.

    Args:
        env: 환경 객체.
        z_threshold: Z방향 거리 임계값(미터). 기본값은 0.144m.
        plane_x_size: 평면의 X방향 크기(미터). 기본값은 0.18m.
        plane_y_size: 평면의 Y방향 크기(미터). 기본값은 0.262m.
        robot_cfg: 로봇 설정. 기본값은 SceneEntityCfg("robot").
        object_cfg: 물체 설정. 기본값은 SceneEntityCfg("object").
        klt_cfg: KLT 설정. 기본값은 SceneEntityCfg("small_klt").

    Returns:
        물체가 평면 영역 위에 위치하고 Z방향 거리가 임계값 이하인지 여부.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    klt: RigidObject = env.scene[klt_cfg.name]

    # 목표 위치
    des_pos_w = klt.data.root_pos_w[:, :3]  # 박스 바닥면이 아니라 지면과 맡닿은 곳으로 부터 58.8mm 위에 위치
    print(f"des_pos: {des_pos_w}")

    # 물체 위치
    obj_pos_w = object.data.root_pos_w[:, :3]
    print(f"obj_pos: {obj_pos_w}")
    
    # 평면 경계 계산 (평면 중심에서 각 방향으로의 반경)
    half_x = plane_x_size / 2.0
    half_y = plane_y_size / 2.0
    
    # 평면의 XY 범위 내에 물체가 있는지 확인
    x_in_bounds = (obj_pos_w[:, 0] >= (des_pos_w[:, 0] - half_x)) & (obj_pos_w[:, 0] <= (des_pos_w[:, 0] + half_x))
    y_in_bounds = (obj_pos_w[:, 1] >= (des_pos_w[:, 1] - half_y)) & (obj_pos_w[:, 1] <= (des_pos_w[:, 1] + half_y))
    xy_in_bounds = x_in_bounds & y_in_bounds
    
    # Z방향 거리 계산 (물체가 평면보다 위에 있는지)
    z_distance = torch.abs(obj_pos_w[:, 2] - des_pos_w[:, 2])
    z_close_enough = (z_distance < z_threshold) & (z_distance > 0)
    
    # 평면 XY 범위 내에 있고 Z방향 거리가 임계값 이하인지 확인
    return xy_in_bounds & z_close_enough