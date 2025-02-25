import zarr
import numpy as np
from pathlib import Path

def check_zarr(zarr_path):
    print(f"\n{'='*100}")
    print(f"Checking {zarr_path}")
    print(f"{'='*100}")
    
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
        
        # 첫 번째 데이터 샘플 출력
        if key == 'img':
            print(f"Image shape: {array[0].shape}")
        elif key == 'point_cloud':
            print(f"Point cloud shape: {array[0].shape}")
        else:
            print("First element:")
            print(array[0])
    
    print("\n[Meta Group]")
    for key in meta_group.keys():
        array = meta_group[key]
        print(f"\nArray: {key}")
        print(f"Shape: {array.shape}")
        print(f"Data: {array[:]}")

# 모든 zarr 파일 확인
zarr_files = [
    'drill_40demo_1024.zarr',
    'dumpling_new_40demo_1024.zarr',
    'pour_40demo_1024.zarr',
    'roll_40demo_1024.zarr'
]

base_path = Path('/home/bak/Projects/3D-Diffusion-Policy/3D-Diffusion-Policy/data')

for zarr_file in zarr_files:
    zarr_path = base_path / zarr_file
    check_zarr(zarr_path)