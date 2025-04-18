# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth, create_pointcloud_from_rgbd

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


"""
Root state.
"""


def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


"""
Joint state.
"""


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


"""
Sensors.
"""


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset


def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.body_ids]
    return link_incoming_forces.view(env.num_envs, -1)


def imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w


def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


def imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]
    # TODO: instance_segmentation_fast 정보를  잘 조절해서 가져오면 될거 같은데.
    # TODO: pick_place_env_cfg.py에서 어떤 instance를 받아올 것인지 argument으로 넘겨주면 될듯.
    # 이 코드에서는 넘겨받은 것을 통해 어떤 instance segmentation을 가져올 것인지 결정함.

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()


def visualize_torch_image(img_tensor, title="Torch Image", batch_idx=0):
    """
    torch 텐서 형태의 이미지를 시각화합니다.
    
    Args:
        img_tensor: torch.Size([batch, height, width, channels]) 형태의 텐서
        title: 표시할 제목
        batch_idx: 여러 이미지 중 표시할 이미지의 인덱스
    """
    import numpy as np  # numpy 임포트 추가
    import matplotlib.pyplot as plt  # matplotlib 임포트 추가
    
    # GPU -> CPU, torch -> numpy 변환
    if img_tensor.is_cuda:
        img_tensor = img_tensor.cpu()
    
    # 첫 번째(혹은 지정된) 배치의 이미지만 선택
    img = img_tensor[batch_idx].numpy()
    
    # 값 범위 확인
    min_val = img.min()
    max_val = img.max()
    
    # 이미지 데이터 타입과 범위에 따른 처리
    if img.dtype == np.float32 or img.dtype == np.float64:
        if max_val > 1.0 or min_val < 0.0:
            print(f"Warning: Image values outside [0,1] range. Min: {min_val}, Max: {max_val}")
            # 0-1 범위로 정규화
            img = (img - min_val) / (max_val - min_val) if max_val > min_val else img
        # 음수값이 있으면 (이미 정규화된 경우) 0-1 범위로 변환
        elif min_val < 0:
            img = (img + 1) / 2  # -1,1 -> 0,1 변환
    
    # RGB 이미지 표시
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"{title} (Shape: {img.shape})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 이미지 정보 출력
    print(f"Image shape: {img.shape}")
    print(f"Value range: [{img.min():.4f}, {img.max():.4f}]")
    print(f"Data type: {img.dtype}")


def apply_rgba_mask(rgb_image, depth_image, seg_image, target_rgba_values):
    """
    RGB 이미지와 depth 이미지에서 특정 RGBA 값에 해당하는 영역만 추출하고 
    나머지 부분은 제거하는 함수
    
    Args:
        rgb_image: RGB 이미지 텐서 [H, W, 3]
        depth_image: Depth 이미지 텐서 [H, W] 또는 [H, W, 1]
        target_rgba_values: 마스크로 사용할 RGBA 값 리스트 [(R, G, B, A), ...]
        seg_image: 세그멘테이션 이미지 텐서 [H, W, 3] (선택적)
        
    Returns:
        masked_rgb: 마스킹된 RGB 이미지
        masked_depth: 마스킹된 depth 이미지
    """
    
    # 장치 및 데이터 타입 정보 가져오기
    device = rgb_image.device
    dtype = rgb_image.dtype
    
    # # 이미지 정보 출력
    # print(f"RGB 이미지 shape: {rgb_image.shape}")
    # print(f"Depth 이미지 shape: {depth_image.shape}")
    # print(f"세그멘테이션 이미지 shape: {seg_image.shape}")
    
    # 마스크 초기화
    height, width = rgb_image.shape[0], rgb_image.shape[1]
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)    
    
    # 세그멘테이션 이미지의 최대값 확인 (정규화 여부)
    seg_max_val = seg_image.max().item()
    is_normalized = seg_max_val <= 1.1
    # print(f"세그멘테이션 이미지 최대값: {seg_max_val}, 정규화 여부: {is_normalized}")
    
    # 타겟 RGBA 값 기반 마스크 생성
    for rgba in target_rgba_values:
        r, g, b, _ = rgba  # alpha 무시
        
        # 정규화된 이미지라면 타겟 값도 정규화
        if is_normalized:
            r_target = r / 255.0
            g_target = g / 255.0
            b_target = b / 255.0
        else:
            r_target = r
            g_target = g
            b_target = b
        
        # 색상 매칭 임계값 (정규화 여부에 따라 조정)
        threshold = 0.05 if is_normalized else 10
        
        # 세그멘테이션 이미지에서 색상 매칭
        r_match = torch.abs(seg_image[:, :, 0] - r_target) <= threshold
        g_match = torch.abs(seg_image[:, :, 1] - g_target) <= threshold
        b_match = torch.abs(seg_image[:, :, 2] - b_target) <= threshold
        
        # 모든 채널이 매칭되는 픽셀 찾기
        color_mask = r_match & g_match & b_match
        
        # 전체 마스크에 추가
        mask = mask | color_mask
        # print(f"색상 ({r}, {g}, {b})에 매칭된 픽셀: {color_mask.sum().item()}")

    # 마스크 상태 확인
    total_mask_pixels = mask.sum().item()   # 기능상 없어도 됨. 디버깅 필요할 때 출력하여 사용
    
    # 마스크 채널 확장
    if len(rgb_image.shape) == 3:
        rgb_mask = mask.unsqueeze(-1).expand(-1, -1, 3)
    else:
        rgb_mask = mask
    
    if len(depth_image.shape) == 3:
        depth_mask = mask.unsqueeze(-1)
    else:
        depth_mask = mask
    
    # 마스크 적용
    masked_rgb = torch.zeros_like(rgb_image)
    masked_depth = torch.zeros_like(depth_image)
    
    # 마스크 내 픽셀만 복사
    masked_rgb[rgb_mask] = rgb_image[rgb_mask]
    masked_depth[depth_mask] = depth_image[depth_mask]
    
    return masked_rgb, masked_depth



def point_cloud(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    num_points: int = 1024,
) -> torch.Tensor:
    """Creates point cloud from depth image using camera intrinsics and pose.

    Args:
        env: The environment.
        sensor_cfg: The camera sensor configuration. Defaults to SceneEntityCfg("camera").

    Returns:
        Point cloud in world coordinates. Shape: (num_points, 3)
    """
    # Get camera sensor
    sensor: Camera = env.scene.sensors[sensor_cfg.name]

    # 아래 정보는 argument로 넘겨받아야 함. 일단 구현을 위해 내부에서 직접적으로 언급해봄
    semantics = sensor.data.info[0]['instance_segmentation_fast']['idToSemantics']
    mask_list = []
    for rgba, info in semantics.items():
        # class 값을 가져와서 소문자인지 확인
        class_name = info['class']
        # 모든 글자가 소문자인지 확인 (islower()는 문자열 전체가 소문자일 때만 True 반환)
        if class_name.islower():
            mask_list.append(rgba)
        # if class_name.islower() and class_name != 'robot':
        #     mask_list.append(rgba)

    rgb = sensor.data.output["rgb"][0]
    depth = sensor.data.output["depth"][0]
    seg = sensor.data.output["instance_segmentation_fast"][0]
    mask_rgb, mask_depth = apply_rgba_mask(rgb, depth, seg, mask_list)

    point_cloud, rgb = create_pointcloud_from_rgbd(
        intrinsic_matrix=sensor.data.intrinsic_matrices[0],  # Using first camera
        depth=mask_depth,
        rgb=mask_rgb,
        position=sensor.data.pos_w[0],
        orientation=sensor.data.quat_w_ros[0],
        device=env.device
    )

    # Find unique points and their indices
    unique_pcd, inverse_indices = torch.unique(point_cloud, dim=0, return_inverse=True)

    # 각 unique 포인트가 몇 번 등장했는지 계산 (평균을 위해 나중에 사용됨)
    counts = torch.bincount(inverse_indices, minlength=unique_pcd.shape[0]).unsqueeze(1)

    # 각 고유한 포인트에 해당하는 rgb값 합산
    unique_rgb = torch.zeros((unique_pcd.shape[0], rgb.shape[1]), dtype=rgb.dtype, device=rgb.device)
    unique_rgb.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, rgb.shape[1]), rgb)

    # 평균값으로 변환 (sum / count)
    unique_rgb = unique_rgb / counts.clamp(min=1)

    
    '''
    import open3d as o3d
    import numpy as np
    def visualize_pointcloud_o3d(points: torch.Tensor, color: torch.Tensor = None):
        """Visualize point cloud using Open3D.
        
        Args:
            points: Point cloud tensor of shape (N, 3)
        """
        # Convert to numpy if on GPU
        if points.is_cuda:
            points = points.cpu()
        points = points.numpy()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if color is not None:
            if color.is_cuda:
                color = color.cpu()
            color = color.numpy()
            if np.max(color) > 1:
                color = color / 255.0
            pcd.colors = o3d.utility.Vector3dVector(color)

        # Create a visualizer with a gray background
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray color (RGB)

        vis.run()
        vis.destroy_window()


    visualize_pointcloud_o3d(unique_pcd, unique_rgb)
    visualize_pointcloud_o3d(unique_pcd)
    visualize_pointcloud_o3d(point_cloud, rgb)
    visualize_pointcloud_o3d(sampled_pcd, sampled_rgb)
    '''

    # Uniform random sampling
    if unique_pcd.shape[0] > num_points:
        # Random sampling if we have more points than needed
        perm = torch.randperm(unique_pcd.shape[0], device=env.device)
        sampled_pcd = unique_pcd[perm[:num_points]]
        sampled_rgb = unique_rgb[perm[:num_points]]
    else:
        # If we have fewer unique points than needed        
        # 1. Calculate how many more points we need
        remaining = num_points - unique_pcd.shape[0]
        
        # 2. Add noise to existing points to create new samples
        if remaining > 0:
            # Select points to duplicate with noise (can select same point multiple times)
            idx = torch.randint(unique_pcd.shape[0], (remaining,), device=env.device)
            selected_pcd = unique_pcd[idx]
            selected_rgb = unique_rgb[idx]

            # Add small random noise to create "new" points
            noise = torch.randn_like(selected_pcd) * 0.002  # Small noise
            additional_points = selected_pcd + noise
            additional_rgb = selected_rgb
            
            # Combine original unique points with additional points
            sampled_pcd = torch.cat([unique_pcd, additional_points], dim=0)
            sampled_rgb = torch.cat([unique_rgb, additional_rgb], dim=0)

    # Concatenate sampled point cloud and RGB
    sampled_data = torch.cat([sampled_pcd, sampled_rgb], dim=1)

    # # shape: (num_points, 3)을 (1, num_points, 3)으로 확장. sequence dimension 추가
    return sampled_data.unsqueeze(0)
    # return sampled_data


# 격자기반 point cloud sampling
def adaptive_voxel_downsample(points: torch.Tensor, target_points: int = 1024) -> torch.Tensor:
    """Voxel downsampling with adaptive voxel size to achieve target number of points.
    
    Args:
        points: Input point cloud (N, 3)
        target_points: Desired number of points after downsampling
        
    Returns:
        Downsampled point cloud with approximately target_points points
    """
    # Initialize voxel size based on point cloud bounds
    bounds = torch.max(points, dim=0)[0] - torch.min(points, dim=0)[0]
    voxel_size = torch.max(bounds) / (target_points ** (1/3))  # Initial estimate
    
    # Binary search for appropriate voxel size
    max_iter = 10
    min_size = 0
    max_size = voxel_size * 2
    
    for _ in range(max_iter):
        # Quantize points to voxel indices
        voxels = torch.floor(points / voxel_size)
        
        # Get unique voxels
        unique_voxels = torch.unique(voxels, dim=0)
        num_voxels = len(unique_voxels)
        
        # Adjust voxel size based on number of points
        if num_voxels > target_points * 1.1:  # Too many points
            min_size = voxel_size
            voxel_size = (voxel_size + max_size) / 2
        elif num_voxels < target_points * 0.9:  # Too few points
            max_size = voxel_size
            voxel_size = (min_size + voxel_size) / 2
        else:  # Close enough
            break
    
    # Final downsampling with found voxel size
    voxels = torch.floor(points / voxel_size)
    _, inverse_indices = torch.unique(voxels, dim=0, return_inverse=True)
    
    # Calculate centroids for each voxel
    downsampled = torch.zeros((len(torch.unique(inverse_indices)), 3), device=points.device)
    for i in range(len(downsampled)):
        mask = inverse_indices == i
        downsampled[i] = points[mask].mean(dim=0)
    
    return downsampled


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,  # we pre-process based on model
        )
        # store the device of the image
        image_device = image_data.device
        # forward the images through the model
        features = self._inference_fn(self._model, image_data, **(inference_kwargs or {}))

        # move the features back to the image device
        return features.detach().to(image_device)

    """
    Helper functions.
    """

    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}


"""
Actions.
"""


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)



"""
End-effector.
"""


def end_effector_pose(
    env: ManagerBasedEnv,
    articulation_name: str = "robot",
    end_effector_name: str = "panda_hand",  # 로봇 모델의 end-effector 링크 이름
    body_offset=None  # offset 추가 가능
) -> torch.Tensor:
    """Gets the end-effector pose (position and orientation).
    
    이 함수는 task_space_actions.py의 _compute_frame_pose와 동일한 방식으로
    end-effector의 pose를 계산합니다.

    Args:
        env: The environment.
        articulation_name: The name of the articulation. Defaults to "robot".
        end_effector_name: The name of the end-effector link. Defaults to "panda_hand".
        body_offset: Optional offset to apply to the body pose.

    Returns:
        A tensor of shape (1, 7) representing position (3) and quaternion (4).
    """
    # Get the robot articulation
    articulation: Articulation = env.scene.articulations[articulation_name]

    # Get the body ID for the end-effector
    body_ids, _ = articulation.find_bodies(end_effector_name)
    if len(body_ids) == 0:
        raise ValueError(f"Body '{end_effector_name}' not found in articulation '{articulation_name}'")
    body_idx = body_ids[0]
    
    # Obtain quantities from simulation (task_space_actions.py의 _compute_frame_pose와 동일한 로직)
    ee_pos_w = articulation.data.body_pos_w[:, body_idx]
    ee_quat_w = articulation.data.body_quat_w[:, body_idx]
    root_pos_w = articulation.data.root_pos_w
    root_quat_w = articulation.data.root_quat_w
    
    # Compute the pose of the body in the root frame
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    
    # Account for the offset (body_offset이 제공된 경우)
    if body_offset is not None:
        offset_pos = torch.tensor(body_offset.pos, device=env.device).repeat(env.num_envs, 1)
        offset_rot = torch.tensor(body_offset.rot, device=env.device).repeat(env.num_envs, 1)
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, offset_pos, offset_rot
        )
    
    # Combine position and orientation into a single pose tensor (7D)
    ee_pose = torch.cat([ee_pose_b[0], ee_quat_b[0]])
    
    return ee_pose  # Shape: (7), 3 position + 4 quaternion


def ee_pose_and_griper_pos(
        env: ManagerBasedEnv,
        articulation_name: str = "robot",
        end_effector_name: str = "panda_hand",  # 로봇 모델의 end-effector 링크 이름
        body_offset=None  # offset 추가 가능
    ) -> torch.Tensor:
    """Gets the end-effector pose (position and orientation) and joint positions.

    Args:
        env: The environment.
        articulation_name: The name of the articulation. Defaults to "robot".
        end_effector_name: The name of the end-effector link. Defaults to "panda_hand".
        body_offset: Optional offset to apply to the body pose.

    Returns:
        A tensor of shape (1, 7) representing position (3) and quaternion (4).
    """

    # # Get the robot articulation
    # articulation: Articulation = env.scene.articulations[articulation_name]
    # gripper_joint_pos = articulation.data.joint_pos[:, -2:]  # 마지막 2개의 joint만 사용 (gripper joint).
    gripper_joint_pose = joint_pos(env, asset_cfg=SceneEntityCfg(articulation_name))[:, -2:] # 위에 주석처리 한 줄과 동일한 결과
    gripper_joint_pose = gripper_joint_pose[0]   # ee_pose와 차원을 맞추기 위해 squeeze
    ee_pose = end_effector_pose(env, articulation_name, end_effector_name, body_offset)

    ee_pose_and_gripper_pos = torch.cat([ee_pose, gripper_joint_pose])
    ee_pose_and_gripper_pos = ee_pose_and_gripper_pos.unsqueeze(0)  # (1, 9)로 변환
    
    return ee_pose_and_gripper_pos
