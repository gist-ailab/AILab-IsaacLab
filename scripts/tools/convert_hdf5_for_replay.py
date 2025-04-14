import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import torch
import shutil
from isaaclab.utils.math import quat_box_minus, normalize, wrap_to_pi, euler_xyz_from_quat


def quat_from_angle_axis(angle, axis):
    """
    각도-축 표현을 쿼터니언으로 변환합니다.
    
    Args:
        angle: 회전 각도(라디안). 
        axis: 정규화된 회전축.
        
    Returns:
        (w, x, y, z) 형식의 쿼터니언.
    """
    half_angle = angle / 2.0
    sin_half_angle = torch.sin(half_angle)
    
    w = torch.cos(half_angle)
    x = axis[:, 0] * sin_half_angle
    y = axis[:, 1] * sin_half_angle
    z = axis[:, 2] * sin_half_angle
    
    return torch.stack([w, x, y, z], dim=1)

def quaternion_to_euler_xyz_diff(q_prev, q_curr, scale_factor=11.0):
    """
    두 쿼터니언 간의 회전 차이를 오일러 각도로 변환합니다.
    
    Args:
        q_prev: 이전 프레임의 쿼터니언 (w, x, y, z) 형식.
        q_curr: 현재 프레임의 쿼터니언 (w, x, y, z) 형식.
        
    Returns:
        오일러 각도 차이 (roll, pitch, yaw).
    """
    # NumPy 배열을 PyTorch 텐서로 변환
    if isinstance(q_prev, np.ndarray):
        q_prev = torch.from_numpy(q_prev).float()
    if isinstance(q_curr, np.ndarray):
        q_curr = torch.from_numpy(q_curr).float()
    
    # CUDA 사용 가능하면 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_prev = q_prev.to(device)
    q_curr = q_curr.to(device)

    # 단일 쿼터니언인 경우 배치 차원 추가
    is_single = q_prev.dim() == 1
    if is_single:
        q_prev = q_prev.unsqueeze(0)
        q_curr = q_curr.unsqueeze(0)

    # 쿼터니언 정규화
    q_prev_norm = normalize(q_prev)
    q_curr_norm = normalize(q_curr)
    
    # 회전 차이 계산 (순서 변경: q_curr에서 q_prev로의 변환)
    rot_vec = quat_box_minus(q_curr_norm, q_prev_norm)  # 순서 변경 - 현재에서 이전으로
    
    # 회전 각도와 축 계산
    angle = torch.norm(rot_vec, dim=1)
    angle = angle * scale_factor    # scale factor 적용
    
    # 초기화
    roll = torch.zeros_like(angle)
    pitch = torch.zeros_like(angle)
    yaw = torch.zeros_like(angle)
    
    # 작은 회전이 아닌 경우만 계산
    non_zero_idx = angle >= 1e-6
    if torch.any(non_zero_idx):
        axis = torch.zeros_like(rot_vec)
        axis[non_zero_idx] = rot_vec[non_zero_idx] / angle[non_zero_idx].unsqueeze(-1)
        
        rot_quat = quat_from_angle_axis(angle, axis)
        r, p, y = euler_xyz_from_quat(rot_quat)
        
        roll[non_zero_idx] = r[non_zero_idx]
        pitch[non_zero_idx] = p[non_zero_idx]
        yaw[non_zero_idx] = y[non_zero_idx]
    
    # -π ~ π 범위로 정규화
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)
    yaw = wrap_to_pi(yaw)

    # 결과 반환
    roll_np = roll.cpu().numpy()
    pitch_np = pitch.cpu().numpy()
    yaw_np = yaw.cpu().numpy()

    if is_single:
        return np.array([roll_np[0], pitch_np[0], yaw_np[0]], dtype=np.float32)
    else:
        return np.stack([roll_np, pitch_np, yaw_np], axis=0).astype(np.float32)

def hdf5_for_replay(hdf5_path, hdf5_replay_path):
    """
    Convert HDF5 file to a replayable format by adding agent position differences
    directly under data/{demo_idx}/actions.
    
    Args:
        hdf5_path (str): Path to the input HDF5 file.
        hdf5_replay_path (str): Path to the output replayable HDF5 file.
    """
    # 원본 파일을 복사
    shutil.copy(hdf5_path, hdf5_replay_path)
    
    # 복사된 HDF5 파일 열기
    with h5py.File(hdf5_replay_path, 'r+') as hdf5_file:
        # 각 데모를 반복
        for demo_idx in tqdm(hdf5_file['data'].keys(), desc="Processing demonstrations"):
            demo = hdf5_file['data'][demo_idx]
            obs_policy = demo['post_obs_policy']
            
            # post_obs_policy에서 필요한 데이터 가져오기
            original_actions = obs_policy['actions'][:]
            agent_pos = obs_policy['agent_pos'][:]
            num_frames = original_actions.shape[0]
            
            # 각 프레임별 agent_pos_diff 계산할 배열 생성 
            # 결과 shape: (num_frames, 7) - [pos_diff(3), rot_diff(3), gripper(1)]
            agent_pos_diffs = np.zeros((num_frames, 7), dtype=np.float32)
            
            # 각 프레임에 대해 위치와 회전 차이 계산
            for frame_idx in range(num_frames):
                if frame_idx == 0:
                    # 첫 프레임은 차이가 없음, 그리퍼 상태만 유지
                    agent_pos_diffs[frame_idx] = np.array([0, 0, 0, 0, 0, 0, original_actions[0][-1]], dtype=np.float32)
                else:
                    scale_factor = 11.0 # 스케일 팩터; frame 간 차이가 너무 작아서 스케일링 필요
                    # 위치 차이 계산
                    pos_diff = agent_pos[frame_idx][:3] - agent_pos[frame_idx-1][:3]
                    pos_diff = pos_diff * scale_factor
                    
                    # 회전 차이 계산
                    q_curr = torch.from_numpy(agent_pos[frame_idx][3:7]).float().cuda()
                    q_prev = torch.from_numpy(agent_pos[frame_idx-1][3:7]).float().cuda()
                    rot_diff = quaternion_to_euler_xyz_diff(q_prev, q_curr, scale_factor)
                    # rot_diff = quaternion_to_euler_xyz_diff(q2, q1)
                    
                    # 그리퍼 상태
                    gripper_condition = original_actions[frame_idx][-1]
                    
                    # 위치 차이, 회전 차이, 그리퍼 상태 결합
                    agent_pos_diffs[frame_idx] = np.hstack((pos_diff, rot_diff, [gripper_condition]))
            
            # demo_idx 바로 아래에 actions 데이터셋 생성 (이미 있으면 삭제 후 재생성)
            if 'actions' in demo:
                del demo['actions']
            
            # 새 actions 데이터셋 생성
            demo.create_dataset('actions', data=agent_pos_diffs)
    
    print(f"Conversion completed. Replayable HDF5 file saved at {hdf5_replay_path}")


def delete_failed_episodes(hdf5_path, inspected_file_path):
    """
    HDF5 파일에서 실패한 에피소드를 삭제합니다.
    
    Args:
        hdf5_path (str): HDF5 파일 경로.
    """

    # 원본 파일을 복사
    shutil.copy(hdf5_path, inspected_file_path)

    failed_indices = [1, 5, 9, 13, 16, 17]

    with h5py.File(inspected_file_path, 'r+') as hdf5_file:
        # 삭제할 키 목록 수집
        keys_to_delete = []
        
        for demo_key in hdf5_file['data'].keys():
            # "demo_숫자" 형식에서 숫자 부분 추출
            try:
                demo_idx = int(demo_key.split('_')[1])
                if demo_idx in failed_indices:
                    keys_to_delete.append(demo_key)
            except (IndexError, ValueError):
                print(f"Warning: Could not parse index from key {demo_key}")
        
        # 수집된 키 목록을 기반으로 데이터 삭제
        for key in keys_to_delete:
            del hdf5_file['data'][key]
            print(f"Deleted failed episode: {key}")
            
        # 남은 데모들의 인덱스 재정렬
        remaining_keys = sorted(list(hdf5_file['data'].keys()), 
                              key=lambda x: int(x.split('_')[1]))
        
        # 임시 데이터 저장을 위한 그룹 생성
        if 'temp' in hdf5_file:
            del hdf5_file['temp']
        hdf5_file.create_group('temp')
        
        # 기존 데이터를 임시 그룹으로 복사
        for i, old_key in enumerate(remaining_keys):
            hdf5_file.copy(f'data/{old_key}', f'temp/{old_key}')
        
        # 기존 data 그룹 삭제 후 새로 생성
        del hdf5_file['data']
        hdf5_file.create_group('data')
        
        # 임시 그룹에서 데이터를 다시 data 그룹으로 복사 (새 인덱스 부여)
        for i, old_key in enumerate(remaining_keys):
            new_key = f'demo_{i}'
            hdf5_file.copy(f'temp/{old_key}', f'data/{new_key}')
            print(f"Renumbered: {old_key} -> {new_key}")
        
        # 임시 그룹 삭제
        del hdf5_file['temp']
        
    print(f"성공적으로 {len(keys_to_delete)}개의 실패 에피소드를 삭제하고, 남은 데모의 인덱스를 재정렬했습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDF5 파일을 시뮬레이터 재생용 형식으로 변환합니다.")
    parser.add_argument("--hdf5_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/240414_pick_place.hdf5',
                        help="입력 HDF5 파일 경로")
    parser.add_argument("--converted_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/240414_pick_place_replay.hdf5',
                        help="출력 재생용 HDF5 파일 경로")
    parser.add_argument("--inspected_file_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/240414_pick_place_inspected.hdf5',
                        help="검사된 HDF5 파일 경로")

    args = parser.parse_args()

    # # HDF5 재생용 변환
    # try:
    #     hdf5_for_replay(args.hdf5_path, args.converted_path)
    # except Exception as e:
    #     print(f"변환 중 오류 발생: {e}")
    #     import traceback
    #     traceback.print_exc()

    delete_failed_episodes(args.converted_path, args.inspected_file_path)