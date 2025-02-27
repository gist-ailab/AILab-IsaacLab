import os
import h5py
import zarr
import numpy as np
import argparse
import shutil
from tqdm import tqdm
from termcolor import cprint

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
            # initial_state_group = demo_group['initial_state']
            # articulation_group = initial_state_group['articulation']
            # robot_group = articulation_group['robot']
            # joint_position = robot_group['joint_position'][:]
            # joint_velocity = robot_group['joint_velocity'][:]
            # root_pose = robot_group['root_pose'][:]

            ee_pose = obs_group['end_effector_pose'][:]
            ee_pose = np.squeeze(ee_pose, axis=1)
            joint_pos = obs_group['joint_pos'][:]
            gripper_joint_pos = joint_pos[:, -2:]

            # State : ee_pose + gripper_joint_pos
            # state = np.concatenate(
            #     (np.tile(ee_pose, [len(actions), 1]),
            #      np.tile(gripper_joint_pos, [len(actions), 1])), axis=1)
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
            point_cloud_arrays.append(point_cloud)

            # Update total count and store episode end index
            total_count += len(actions)
            episode_ends_arrays.append(total_count)

    # Concatenate all arrays
    action_arrays = np.concatenate(action_arrays, axis=0)
    state_arrays = np.concatenate(state_arrays, axis=0)
    img_arrays = np.concatenate(img_arrays, axis=0)
    img_arrays = np.transpose(img_arrays, (0, 2, 1, 3)) # Change image channel from (N,W,H,C) -> (N,H,W,C)
    point_cloud_arrays = np.concatenate(point_cloud_arrays, axis=0)
    depth_arrays = np.concatenate(depth_arrays, axis=0)
    depth_arrays = np.transpose(depth_arrays, (0, 2, 1, 3)) # Change image channel from (N,W,H,C) -> (N,H,W,C)
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
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/zzz.hdf5',
                        help="Path to the input HDF5 file.")
    parser.add_argument("--zarr_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/zzz.zarr',
                        help="Path to the output Zarr file.")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Chunk size for Zarr arrays (default: 100).")
    # parser.add_argument("--overwrite", action="store_true", default=True,
    #                     help="Overwrite the output Zarr file if it exists.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", default=True,
                        help="Do not overwrite the output Zarr file if it exists. (Default: overwrite)")
    args = parser.parse_args()

    # Convert HDF5 to Zarr
    hdf5_to_zarr(args.hdf5_path, args.zarr_path, args.chunk_size)

    # Check the Zarr file (optional)
    check_zarr(args.zarr_path)