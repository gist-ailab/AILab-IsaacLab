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


def visualize_image(img_array):
    """Visualizes an image using matplotlib"""
    plt.figure()
    plt.imshow(img_array)
    plt.axis('off')  # Hide axis
    plt.title("Sample Image")
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

    # Initialize lists to collect all data
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0
    episode_count = 0

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # Iterate through demos
        for demo_name in tqdm(sorted(hdf5_file['data'].keys()), desc=f"Processing {hdf5_path}"):
        # for demo_name in sorted(hdf5_file['data'].keys()):
            episode_count += 1
            demo_group = hdf5_file['data'][demo_name]
            obs_group = demo_group['obs']

            # Actions
            actions = obs_group['actions'][:]
            action_arrays.append(actions)

            # State
            ee_pose = obs_group['end_effector_pose'][:]
            ee_pose = np.squeeze(ee_pose, axis=1)
            joint_pos = obs_group['joint_pos'][:]
            gripper_joint_pos = joint_pos[:, -2:]

            # State : ee_pose + gripper_joint_pos
            state = np.concatenate((ee_pose, gripper_joint_pos), axis=1)
            state_arrays.append(state.astype(np.float32))

            # Vision data           
            # Get RGB image data
            rgb_image = obs_group['rgb_image'][:]  # Should be shape (num_frames, H, W, 3)
            img_arrays.append(rgb_image)
            # Get depth data
            depth_image = obs_group['depth_image'][:]
            depth_arrays.append(depth_image)
            # Get point cloud data
            point_cloud = obs_group['point_cloud'][:]
            # sampled_pcd = sample_rgb_point_cloud(point_cloud, num_points=1024)
            point_cloud_arrays.append(point_cloud)

            # Update total count and store episode end index
            total_count += len(actions)
            episode_ends_arrays.append(total_count)

    # Concatenate all arrays
    action_arrays = np.concatenate(action_arrays, axis=0)
    state_arrays = np.concatenate(state_arrays, axis=0)
    img_arrays = np.concatenate(img_arrays, axis=0)
    point_cloud_arrays = np.concatenate(point_cloud_arrays, axis=0)
    depth_arrays = np.concatenate(depth_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    # Create zarr file
    zarr_root = zarr.group(zarr_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    # Set up compressor
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # Define chunk sizes
    img_chunk_size = (chunk_size, *img_arrays.shape[1:])
    state_chunk_size = (chunk_size, state_arrays.shape[1])
    point_cloud_chunk_size = (chunk_size, *point_cloud_arrays.shape[1:])
    depth_chunk_size = (chunk_size, *depth_arrays.shape[1:])
    action_chunk_size = (chunk_size, action_arrays.shape[1])
    
    print('\n')
    cprint('-' * 100, 'cyan')
    cprint(f'Saving data to {zarr_path}', 'cyan')
    cprint('-' * 100, 'cyan')

    # Create datasets with all data at once
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8',
                             overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32',
                             overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32',
                             overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32',
                             overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32',
                             overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    # Print statistics
    print(f"Total {episode_count} demos")
    print(f"img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]")
    print(f"point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]")
    print(f"depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]")
    print(f"state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]")
    print(f"action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]")
    print(f"Conversion complete. Zarr file saved to {zarr_path}")

    # Visualize sample data if enabled
    if args.visualize:
        print("Visualizing sample data...")
        if len(img_arrays) > 0:
            visualize_image(img_arrays[0])
        if len(point_cloud_arrays) > 0:
            visualize_point_cloud(point_cloud_arrays[0])


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 file to Zarr format.")
    parser.add_argument("--hdf5_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/xyz.hdf5',
                        help="Path to the input HDF5 file.")
    parser.add_argument("--zarr_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/xyz.zarr',
                        help="Path to the output Zarr file.")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Chunk size for Zarr arrays (default: 100).")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", default=True,
                        help="Do not overwrite the output Zarr file if it exists. (Default: overwrite)")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Visualize sample image and point cloud after conversion.")
    args = parser.parse_args()

    # Convert HDF5 to Zarr
    hdf5_to_zarr(args.hdf5_path, args.zarr_path, args.chunk_size)

    # Check the Zarr file (optional)
    check_zarr(args.zarr_path)