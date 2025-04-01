import os
import h5py
import zarr
import numpy as np
import argparse
import shutil
from tqdm import tqdm
from termcolor import cprint
import open3d as o3d  # Import Open3D
import matplotlib.pyplot as plt # Import matplotlib
import torch
from isaaclab.utils.math import quat_from_euler_xyz


def visualize_point_cloud(points):
    """Visualizes a point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Assuming first 3 columns are xyz
    if points.shape[1] > 3:  # If there are more than 3 columns, treat them as colors
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.0) # Normalize to 0-1 for colors
    # o3d.visualization.draw_geometries([pcd])

    # Create a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Create a visualizer with custom settings
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    
    # Get render options and set background color to gray
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray color (RGB)
    
    # Set camera position for better viewing
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)  # Adjust zoom level as needed
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


# def visualize_image(img_array):
#     """Visualizes an image using matplotlib"""
#     plt.figure()
#     # plt.imshow(img_array)
#     plt.imshow(img_array, cmap='gray' if len(img_array.shape) == 2 or img_array.shape[-1] == 1 else None) # Add cmap='gray' for grayscale
#     plt.axis('off')  # Hide axis
#     plt.title("Sample Image")
#     plt.show()


def visualize_image(img_array):
    """
    정규화된 이미지 데이터를 올바르게 시각화하는 함수
    
    Args:
        img_array: 시각화할 이미지 배열 (정규화되었거나 원본)
    """
    plt.figure(figsize=(10, 8))
    
    # 이미지 타입 확인 및 전처리
    if img_array.dtype == np.float32 or img_array.dtype == np.float64:
        # 값 분포 확인 (디버깅용)
        print(f"이미지 값 범위: [{img_array.min()}, {img_array.max()}], 평균: {img_array.mean()}")
        
        # RGB 이미지인지 깊이 이미지인지 확인
        if len(img_array.shape) == 3 and img_array.shape[-1] == 3:  # RGB 이미지
            # 평균이 감산된 이미지 처리 (RGB)
            if img_array.mean() < 0.1:  # 평균 중심화 되었을 가능성 높음
                # 평균 추가하여 복원 (0.5는 일반적인 RGB 평균값)
                display_img = img_array + 0.5
                display_img = np.clip(display_img, 0, 1)
            else:
                # 이미 [0,1] 범위인 경우 그대로 사용
                display_img = np.clip(img_array, 0, 1)
                
        else:  # 깊이 이미지 또는 단일 채널 이미지
            # 무한대 값 처리
            img_clean = img_array.copy()
            img_clean[np.isinf(img_clean)] = 0
            
            # 유효한 값이 있는 경우만 정규화
            non_zero_mask = img_clean > 0
            if non_zero_mask.any():
                min_val = np.min(img_clean[non_zero_mask])
                max_val = np.max(img_clean[non_zero_mask])
                
                if max_val > min_val:
                    # 정규화
                    display_img = np.zeros_like(img_clean)
                    display_img[non_zero_mask] = (img_clean[non_zero_mask] - min_val) / (max_val - min_val)
                else:
                    display_img = img_clean
            else:
                display_img = img_clean
    else:
        # 정수형 이미지 (uint8 등)
        display_img = img_array
    
    # 이미지 타입에 따른 시각화
    if len(display_img.shape) == 2 or (len(display_img.shape) == 3 and display_img.shape[-1] == 1):
        # 단일 채널 이미지 (깊이 등)
        if len(display_img.shape) == 3:
            display_img = display_img.squeeze(-1)  # 마지막 차원 제거
        plt.imshow(display_img, cmap='plasma')
        plt.colorbar(label='Depth')
    else:
        # RGB 이미지
        plt.imshow(display_img)
    
    plt.axis('off')
    plt.title("Visualized Image")
    plt.tight_layout()
    plt.show()


def sample_rgb_point_cloud(point_cloud, num_points=1024, device='cuda'):
    """
    Samples a fixed number of points from the input point cloud.
    Args:
        point_cloud: Input point cloud with shape (N, 6) where N is the number of points.
        num_points: Number of points to sample.
    Returns:
        Sampled point cloud with shape (num_points, 6).
    """
    
    # Convert numpy array to torch tensor if needed
    if isinstance(point_cloud, np.ndarray):
        point_cloud = torch.from_numpy(point_cloud).to(device)

    # Get the shape information
    num_frames, num_points_orig, feat_dim = point_cloud.shape

    # Initialize output tensor
    sampled_point_clouds = []

    # Process each frame
    for frame_idx in tqdm(range(num_frames), desc="Sampling point clouds"):
        # Get current frame's point cloud
        pc = point_cloud[frame_idx]
        
        # Split into position and color
        positions = pc[:, :3]
        colors = pc[:, 3:]
        
        # Find unique points and their indices
        unique_pcd, inverse_indices = torch.unique(positions, dim=0, return_inverse=True)
        
        # Get unique RGB values by averaging colors for each unique position
        unique_rgb = torch.zeros((unique_pcd.shape[0], 3), dtype=colors.dtype, device=colors.device)
        
        for i in range(unique_pcd.shape[0]):
            # i번째 unique 포인트에 해당하는 원본 포인트들의 마스크
            mask = (inverse_indices == i)
            if mask.sum() > 0:
                # 해당 마스크에 해당하는 rgb값들의 평균을 사용
                unique_rgb[i] = colors[mask].mean(dim=0)
        
        # Sample the points to match num_points
        if unique_pcd.shape[0] > num_points:
            # Random sampling if we have more points than needed
            perm = torch.randperm(unique_pcd.shape[0], device=device)
            sampled_pcd = unique_pcd[perm[:num_points]]
            sampled_rgb = unique_rgb[perm[:num_points]]
        else:
            # If we have fewer unique points than needed
            remaining = num_points - unique_pcd.shape[0]
            
            if remaining > 0:
                # Select points to duplicate with noise (can select same point multiple times)
                idx = torch.randint(unique_pcd.shape[0], (remaining,), device=device)
                selected_pcd = unique_pcd[idx]
                selected_rgb = unique_rgb[idx]
                
                # Add small random noise to create "new" points
                noise = torch.randn_like(selected_pcd) * 0.002  # Small noise
                additional_points = selected_pcd + noise
                additional_rgb = selected_rgb
                
                # Combine original unique points with additional points
                sampled_pcd = torch.cat([unique_pcd, additional_points], dim=0)
                sampled_rgb = torch.cat([unique_rgb, additional_rgb], dim=0)
            else:
                # If we already have the exact number of points needed
                sampled_pcd = unique_pcd
                sampled_rgb = unique_rgb
        
        # Concatenate position and color data
        sampled_data = torch.cat([sampled_pcd, sampled_rgb], dim=1)
        
        # Add to results
        sampled_point_clouds.append(sampled_data)
    
    # Stack all frames into a single tensor
    result = torch.stack(sampled_point_clouds, dim=0)
    
    return result


def hdf5_to_zarr(hdf5_path, zarr_path, chunk_size=100):
    """
    Converts an HDF5 file to a Zarr file.

    Args:
        hdf5_path: Path to the HDF5 file.
        zarr_path: Path to the output Zarr file.
        chunk_size: Chunk size for Zarr arrays along the first dimension (time/steps).
    """

    print(f"Converting {hdf5_path} to {zarr_path}")

    # Check if the Zarr file already exists and handle overwriting
    if os.path.exists(args.zarr_path):
        if args.overwrite:
            print(f"Removing existing zarr file at {args.zarr_path}")
            shutil.rmtree(args.zarr_path)
        else:
            print(f"Error: {args.zarr_path} already exists. Use --overwrite to overwrite it.")
            exit(1)

    # Create zarr file and groups
    zarr_root = zarr.group(zarr_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    
    # Set up compressor
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # First scan to get shapes and total size
    total_frames = 0
    episode_ends = []
    
    # Analyze first episode to get data shapes
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        first_demo_name = sorted(hdf5_file['data'].keys())[0]
        first_demo = hdf5_file['data'][first_demo_name]
        first_obs_policy = first_demo['obs_policy']
        first_obs_vision = first_demo['obs_vision']

        
        # Get sample shapes from 10th step of the first demo
        # conert euler to quaternion on action
        sample_action_euler = torch.from_numpy(first_obs_policy['actions'][10]).to(device='cuda')
        quat = quat_from_euler_xyz(sample_action_euler[3], sample_action_euler[4], sample_action_euler[5])
        sample_action = torch.cat([sample_action_euler[:3], quat, sample_action_euler[6:]], dim=0).cpu().numpy()

        sample_state = first_obs_policy['agent_pos'][10]
        sample_rgb = first_obs_vision['rgb_image'][10]
        sample_depth = first_obs_vision['depth'][10]
        sample_pc = first_obs_vision['point_cloud'][10]
        
        # Count total frames and create episode ends list
        for demo_name in tqdm(sorted(hdf5_file['data'].keys()), desc="Counting frames"):
            demo_group = hdf5_file['data'][demo_name]
            obs_policy_group = demo_group['obs_policy']
            frames = len(obs_policy_group['actions']) -5   # Skip first 5 frames. A margin to remove frames corresponding to the previous episode.
            total_frames += frames
            episode_ends.append(total_frames)
    
    # Create datasets with known shapes and total size
    img_ds = zarr_data.create_dataset('img', 
                                      shape=(total_frames, *sample_rgb.shape),
                                      chunks=(chunk_size, *sample_rgb.shape),
                                      dtype='uint8',
                                      compressor=compressor)
    
    state_ds = zarr_data.create_dataset('state', 
                                        shape=(total_frames, len(sample_state)),
                                        chunks=(chunk_size, len(sample_state)),
                                        dtype='float32',
                                        compressor=compressor)
    
    point_cloud_ds = zarr_data.create_dataset('point_cloud', 
                                             shape=(total_frames, *sample_pc.shape),
                                             chunks=(chunk_size, *sample_pc.shape),
                                             dtype='float32',
                                             compressor=compressor)
    
    depth_ds = zarr_data.create_dataset('depth', 
                                       shape=(total_frames, *sample_depth.shape),
                                       chunks=(chunk_size, *sample_depth.shape),
                                       dtype='float32',
                                       compressor=compressor)
    
    action_ds = zarr_data.create_dataset('action', 
                                        shape=(total_frames, len(sample_action)),
                                        chunks=(chunk_size, len(sample_action)),
                                        dtype='float32',
                                        compressor=compressor)
    
    # Save episode ends metadata
    zarr_meta.create_dataset('episode_ends', 
                            data=np.array(episode_ends),
                            dtype='int64',
                            compressor=compressor)
    
    # Now fill the datasets episode by episode
    current_idx = 0
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        for demo_name in tqdm(sorted(hdf5_file['data'].keys()), desc="Processing episodes"):
            try:
                demo_group = hdf5_file['data'][demo_name]
                obs_policy_group = demo_group['obs_policy']
                obs_vision_group = demo_group['obs_vision']
                
                # Skip first 5 frames [5:]. A margin to remove frames corresponding to the previous episode.

                # Get data
                actions = obs_policy_group['actions'][5:]
                episode_length = len(actions)
                
                # Store actions with quaternion conversion
                actions_tensor = torch.from_numpy(actions).to(device='cuda')
                quat = quat_from_euler_xyz(actions_tensor[:, 3], actions_tensor[:, 4], actions_tensor[:, 5])
                actions = torch.cat([actions_tensor[:, :3], quat, actions_tensor[:, 6:]], dim=1).cpu().numpy()
                action_ds[current_idx:current_idx + episode_length] = actions
                
                # Process and store state
                # ee_pose = obs_group['end_effector_pose'][5:]
                # ee_pose = np.squeeze(ee_pose, axis=1)  # Remove singleton dimension if present
                # joint_pos = obs_group['joint_pos'][5:]
                # gripper_joint_pos = joint_pos[:, -2:]
                # state = np.concatenate((ee_pose, gripper_joint_pos), axis=1)
                state = obs_policy_group['agent_pos'][5:]
                state_ds[current_idx:current_idx + episode_length] = state.astype(np.float32)
                
                # Store RGB image data
                rgb_image = obs_vision_group['rgb_image'][5:] ### TODO: img_ds로 변환할 때, 뭔가 잘못된듯. 검은색 이미지만 저장되고 있다.
                img_ds[current_idx:current_idx + episode_length] = rgb_image
                
                # Store depth data
                depth_image = obs_vision_group['depth'][5:]
                depth_ds[current_idx:current_idx + episode_length] = depth_image
                
                # Store point cloud data
                # point_cloud = obs_group['point_cloud'][:]
                point_cloud = obs_vision_group['point_cloud'][5:]
                point_cloud_ds[current_idx:current_idx + episode_length] = point_cloud
                
                # Update the current index
                current_idx += episode_length
                
            except Exception as e:
                cprint(f"Error processing episode {demo_name}: {e}", "red")
                # Continue with next episode instead of failing
                continue
    
    # Print statistics
    episode_count = len(episode_ends)
    print(f"Total {episode_count} demos, {total_frames} frames successfully processed")
    
    # Print sample data statistics
    print("\nData statistics:")
    img_sample = img_ds[0]
    print(f"img shape: {img_ds.shape}, range: [{np.min(img_sample)}, {np.max(img_sample)}]")
    
    pc_sample = point_cloud_ds[0]
    print(f"point_cloud shape: {point_cloud_ds.shape}, range: [{np.min(pc_sample)}, {np.max(pc_sample)}]")
    
    depth_sample = depth_ds[0]
    print(f"depth shape: {depth_ds.shape}, range: [{np.min(depth_sample)}, {np.max(depth_sample)}]")
    
    state_sample = state_ds[0]
    print(f"state shape: {state_ds.shape}, range: [{np.min(state_sample)}, {np.max(state_sample)}]")
    
    action_sample = action_ds[0]
    print(f"action shape: {action_ds.shape}, range: [{np.min(action_sample)}, {np.max(action_sample)}]")
    
    print(f"Conversion complete. Zarr file saved to {zarr_path}")


    # Visualize sample data if enabled
    if args.visualize:
        print("Visualizing sample data...")
        visualize_image(img_ds[0])  # rgb image가 검게 저장된 문제가 있음.
        visualize_point_cloud(point_cloud_ds[0])


def check_zarr(zarr_path):
    cprint(f"\n{'=' * 100}", 'green')
    cprint(f"Checking {zarr_path}", 'green')
    cprint(f"{'=' * 100}", 'green')

    # zarr 파일 열기
    store = zarr.open(zarr_path, mode='r')

    # 데이터 그룹 확인
    data_group = store['data']
    meta_group = store['meta']

    print("\n[Data Group]")
    for key in data_group.keys():
        array = data_group[key]
        print(f"\nArray: {key}")
        print(f"Shape: {array.shape}")
        print(f"Dtype: {array.dtype}")
        print(f"Chunks: {array.chunks}")

    print("\n[Meta Group]")
    for key in meta_group.keys():
        array = meta_group[key]
        print(f"\nArray: {key}")
        print(f"Shape: {array.shape}")
        print(f"Data: {array[:]}")


def visualize_episode_data(zarr_path, episode_idx=0):
    """
    특정 에피소드의 state(joint position)와 action 데이터를 시각화합니다.
    
    Args:
        zarr_path: Zarr 파일 경로
        episode_idx: 시각화할 에피소드 인덱스 (기본값: 0)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import zarr
    
    # Zarr 파일 열기
    store = zarr.open(zarr_path, mode='r')
    data = store['data']
    meta = store['meta']
    episode_ends = meta['episode_ends'][:]
    
    # 에피소드 시작과 끝 인덱스 계산
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    
    # 에피소드 데이터 추출
    state_data = data['state'][start_idx:end_idx]
    action_data = data['action'][start_idx:end_idx]
    
    # 타임스텝 배열 생성
    time_steps = np.arange(len(state_data))
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Episode {episode_idx} Data Visualization', fontsize=16)
    
    # State 데이터 시각화
    ax1 = axes[0]
    for i in range(state_data.shape[1]):
        if i < state_data.shape[1] - 2:  # End-effector pose
            if i < 3:  # Position
                label = f'EE Pos {i}'
                linestyle = '-'
            else:  # Orientation
                label = f'EE Quat {i - 3}'
                linestyle = '-'
        else:  # Gripper joint position
            label = f'Gripper {i - (state_data.shape[1] - 2)}'
            linestyle = '--'
            
        ax1.plot(time_steps, state_data[:, i], label=label, linestyle=linestyle)
    
    ax1.set_ylabel('State Value')
    ax1.set_title('State (End-effector pose + Gripper joint positions)')
    ax1.grid(True)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1)
    
    # Action 데이터 시각화
    ax2 = axes[1]
    for i in range(action_data.shape[1]):
        ax2.plot(time_steps, action_data[:, i], label=f'Action {i}')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Action Value')
    ax2.set_title('Actions')
    ax2.grid(True)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.85)
    plt.show(block=True)
    
    # 통계 정보 출력
    print(f"\nEpisode {episode_idx} Statistics:")
    print(f"Length: {len(state_data)} time steps")
    print("\nState Statistics:")
    print(f"Shape: {state_data.shape}")
    print(f"Mean: {np.mean(state_data, axis=0)}")
    print(f"Min: {np.min(state_data, axis=0)}")
    print(f"Max: {np.max(state_data, axis=0)}")
    
    print("\nAction Statistics:")
    print(f"Shape: {action_data.shape}")
    print(f"Mean: {np.mean(action_data, axis=0)}")
    print(f"Min: {np.min(action_data, axis=0)}")
    print(f"Max: {np.max(action_data, axis=0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 file to Zarr format.")
    parser.add_argument("--hdf5_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/dummy.hdf5',
                        help="Path to the input HDF5 file.")
    parser.add_argument("--zarr_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/dummy.zarr',
                        help="Path to the output Zarr file.")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Chunk size for Zarr arrays (default: 100).")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", default=True,
                        help="Do not overwrite the output Zarr file if it exists. (Default: overwrite)")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Visualize sample image and point cloud after conversion.")
    parser.add_argument("--visualize_episode", action="store_true", default=True,
                        help="Visualize state and action data for a specific episode.")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Episode index to visualize (default: 0).")
    args = parser.parse_args()

    # Convert HDF5 to Zarr
    hdf5_to_zarr(args.hdf5_path, args.zarr_path, args.chunk_size)

    # Check the Zarr file (optional)
    check_zarr(args.zarr_path)

    # Visualize episode data if requested
    if args.visualize_episode:
        visualize_episode_data(args.zarr_path, args.episode_idx)