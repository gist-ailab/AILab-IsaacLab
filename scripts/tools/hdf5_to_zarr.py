import h5py
import zarr
import numpy as np
import os
import cv2
from termcolor import cprint


def explore_group(group, indent=0):
    # Loop through each item in the group (it could be a dataset or a subgroup)
    for key in group.keys():
        item = group[key]
        # Print the name of the item (dataset or subgroup) with indentation to show the structure
        print('  ' * indent + f"Name: {key}, Type: {type(item)}")
        
        # If the item is a subgroup, recursively explore it
        if isinstance(item, h5py.Group):
            explore_group(item, indent + 1)

def visualize_image_data(data, window_name="Image"):
    """Visualize image data using OpenCV."""
    if len(data.shape) == 4:  # batch of images
        for i, img in enumerate(data):
            if img.shape[0] == 3:  # if channels first (C, H, W)
                img = np.transpose(img, (1, 2, 0))
            
            # Normalize if needed
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            
            cv2.imshow(f"{window_name}_{i}", img)
    else:  # single image
        img = data
        if img.shape[0] == 3:  # if channels first (C, H, W)
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
            
        cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_hdf5_to_zarr(hdf5_path, zarr_path):
    """Convert HDF5 dataset to Zarr format."""
    
    if os.path.exists(zarr_path):
        cprint(f'Data already exists at {zarr_path}', 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        # user_input = input("Do you want to overwrite? (y/n): ")
        user_input = 'y'
        if user_input.lower() == 'y':
            cprint(f'Overwriting {zarr_path}', 'red')
            os.system(f'rm -rf {zarr_path}')
        else:
            cprint('Exiting', 'red')
            return
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    
    # Open HDF5 file and convert
    with h5py.File(hdf5_path, 'r') as hdf:
        # explore_group(hdf)
        # Create zarr groups
        zarr_root = zarr.group(zarr_path)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        
        ### a = hdf['data']['demo_0']['obs']['point_cloud']

        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # 포인트 클라우드 데이터를 x, y, z로 분리
        x = point_cloud_data[:, 0]
        y = point_cloud_data[:, 1]
        z = point_cloud_data[:, 2]

        # 3D 플롯 생성
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 포인트 클라우드 데이터 시각화
        ax.scatter(x, y, z, c='b', marker='o', s=10)  # 'b'는 파란색, s는 마커 크기

        # 축 레이블 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 플롯 표시
        plt.show()
        '''

        # Collect all episode data
        states = []
        actions = []
        episode_ends = []
        total_steps = 0
        
        # Process each episode
        for episode_name in hdf['episodes']:
            episode = hdf[f'episodes/{episode_name}']
            
            # Extract data
            if 'states' in episode:
                states.extend(episode['states'][:])
            if 'actions' in episode:
                actions.extend(episode['actions'][:])
            
            # Track episode end
            if 'actions' in episode:
                total_steps += len(episode['actions'])
                episode_ends.append(total_steps)
        
        # Convert to numpy arrays
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        episode_ends = np.array(episode_ends)
        
        # Set compression
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        
        # Define chunk sizes
        state_chunk_size = (100, states.shape[1]) if len(states) > 0 else None
        action_chunk_size = (100, actions.shape[1]) if len(actions) > 0 else None
        
        # Save arrays
        zarr_data.create_dataset('state', data=states, 
                               chunks=state_chunk_size,
                               dtype='float32',
                               compressor=compressor)
        zarr_data.create_dataset('action', data=actions,
                               chunks=action_chunk_size,
                               dtype='float32',
                               compressor=compressor)
        zarr_meta.create_dataset('episode_ends', data=episode_ends,
                               dtype='int64',
                               compressor=compressor)
        
        # Print information
        cprint('-'*50, 'cyan')
        cprint(f'state shape: {states.shape}, range: [{np.min(states)}, {np.max(states)}]', 'green')
        cprint(f'action shape: {actions.shape}, range: [{np.min(actions)}, {np.max(actions)}]', 'green')
        cprint(f'episode_ends shape: {episode_ends.shape}', 'green')
        cprint(f'Saved zarr file to {zarr_path}', 'green')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, 
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/dummy.hdf5',
                        help="Input HDF5 file path")
    parser.add_argument("--output", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/dummy.zarr',
                        help="Output Zarr directory path")
    args = parser.parse_args()
    
    convert_hdf5_to_zarr(args.input, args.output)