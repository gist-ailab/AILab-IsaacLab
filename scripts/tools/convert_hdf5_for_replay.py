import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import torch
import shutil
import tempfile
import json
from termcolor import cprint
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


def delete_failed_episodes(hdf5_path, inspected_file_path, failed_keys=None):
    """
    HDF5 파일에서 실패한 에피소드를 삭제하고 남은 데모의 인덱스를 재정렬합니다.
    원본 파일의 압축 특성과 속성(attributes)을 유지합니다.
    """

    if failed_keys is None:
        cprint(f'Failed indices가 존재하지 않습니다.', color='red')
        exit()

    # 원본 파일을 복사
    cprint(f'원본 파일을 검사된 파일로 복사 중: {hdf5_path} -> {inspected_file_path}', color='yellow')
    shutil.copy(hdf5_path, inspected_file_path)
    
    print(f"삭제할 실패 에피소드 인덱스: {failed_keys}")

    with h5py.File(inspected_file_path, 'r+') as hdf5_file:        
        # 데이터 삭제
        for key in failed_keys:
            del hdf5_file['data'][key]
            print(f"실패 에피소드 삭제됨: {key}")
        
        # 남은 데모 인덱스 재정렬
        remaining_keys = sorted(list(hdf5_file['data'].keys()), 
                              key=lambda x: int(x.split('_')[1]))
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # 임시 파일에 데이터 복사
        with h5py.File(temp_path, 'w') as temp_hdf5:
            # 기본 구조 복사 (data 그룹 제외)
            for key in hdf5_file.keys():
                if key != 'data':
                    hdf5_file.copy(key, temp_hdf5)
            
            # data 그룹 생성 및 속성 복사
            data_group = temp_hdf5.create_group('data')
            
            # 중요: data 그룹의 속성 복사
            for attr_name, attr_value in hdf5_file['data'].attrs.items():
                data_group.attrs[attr_name] = attr_value
            
            # 남은 데모 재정렬하여 복사 (원본 압축 속성 유지)
            for i, old_key in enumerate(remaining_keys):
                new_key = f'demo_{i}'
                print(f"인덱스 재정렬: {old_key} -> {new_key}")
                
                # 그룹 생성하고 전체 구조 복사
                source_group = hdf5_file['data'][old_key]
                target_group = data_group.create_group(new_key)
                
                # 그룹 속성 복사
                for attr_name, attr_value in source_group.attrs.items():
                    target_group.attrs[attr_name] = attr_value
                
                # 데이터셋과 그룹 복사 (원본 속성 유지)
                for name, item in source_group.items():
                    if isinstance(item, h5py.Group):
                        # 하위 그룹 복사
                        source_group.copy(name, target_group)
                    else:
                        # 데이터셋 복사 (원본 압축 유지)
                        data = item[()]
                        target_group.create_dataset(name, data=data)

    # 임시 파일을 최종 파일로 이동
    shutil.move(temp_path, inspected_file_path)
    
    # 파일 크기 확인
    file_size_mb = os.path.getsize(inspected_file_path) / (1024 * 1024)
    
    print(f"성공적으로 {len(failed_keys)}개의 실패 에피소드를 삭제하고, 남은 {len(remaining_keys)}개 데모의 인덱스를 재정렬했습니다.")
    print(f"최종 파일 크기: {file_size_mb:.2f} MB")


def combine_episodes(inspected_file_path, supplement_hdf5_path, output_file_path=None):
    """
    검사된 HDF5 파일에 추가 교시 데이터를 합친 새 파일을 생성합니다.
    
    Args:
        inspected_file_path: 입력 파일 경로
        supplement_hdf5_path: 추가 데이터 파일 경로
        output_file_path: 출력 파일 경로. None이면 input_file_path에 _final 접미사 추가
    """
    # 출력 파일 경로가 지정되지 않은 경우 _final 접미사 추가
    if output_file_path is None:
        file_name, file_ext = os.path.splitext(inspected_file_path)
        output_file_path = f"{file_name}_final{file_ext}"
    
    # 입력 파일을 출력 파일로 복사하여 시작
    shutil.copy(inspected_file_path, output_file_path)
    
    try:
        with h5py.File(output_file_path, 'r+') as target_file:
            # 기존 파일의 마지막 데모 인덱스 찾기
            existing_keys = list(target_file['data'].keys())
            if not existing_keys:
                last_idx = -1
            else:
                last_idx = max([int(k.split('_')[1]) for k in existing_keys])
            
            print(f"기존 데모 수: {len(existing_keys)}, 마지막 인덱스: {last_idx}")
            
            # 추가 데이터 파일 처리
            with h5py.File(supplement_hdf5_path, 'r') as src_file:
                demo_count = len(src_file['data'].keys())
                print(f"추가할 데모 수: {demo_count}")
                
                for demo_key in tqdm(src_file['data'].keys(), desc="추가 데모 병합"):
                    new_idx = last_idx + 1
                    last_idx += 1
                    new_key = f'demo_{new_idx}'
                    
                    # 그룹 전체를 복사
                    src_file.copy(f'data/{demo_key}', target_file['data'], name=new_key)
                    print(f"추가됨: {demo_key} -> {new_key}")
                
                print(f"총 {demo_count}개의 데모를 성공적으로 병합했습니다.")
                print(f"최종 데모 수: {len(target_file['data'].keys())}")
                
        # 파일 크기 확인
        file_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
        print(f"최종 파일 크기: {file_size_mb:.2f} MB")
        
        return output_file_path
        
    except Exception as e:
        print(f"데이터 병합 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDF5 파일을 시뮬레이터 재생용 형식으로 변환합니다.")
    parser.add_argument("--hdf5_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/240415_pick_place.hdf5',
                        help="입력 HDF5 파일 경로")
    parser.add_argument("--converted_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/240415_pick_place_replay.hdf5',
                        # default='/home/bak/Projects/AILab-IsaacLab/datasets/240415_pick_place_replay_inspected.hdf5',
                        help="출력 재생용 HDF5 파일 경로 (기본값: 입력 파일명에 _replay 추가)")
    parser.add_argument("--inspected_file_path", type=str,
                        default=None, 
                        help="검사된 HDF5 파일 경로 (기본값: 재생용 파일명에 _inspected 추가)")
    parser.add_argument("--supplement_hdf5_path", type=str,
                        default='/home/bak/Projects/AILab-IsaacLab/datasets/240415_pick_place_supplement.hdf5',
                        help="추가 교시 정보 HDF5 파일 경로")
    parser.add_argument("--final_path", type=str,
                        default=None,
                        help="최종 결과 파일 경로 (기본값: 검사된 파일명에 _final 추가)")
    # parser.add_argument("--failed_indices", type=list, default=['demo_6', 'demo_9', 'demo_10'],
    # parser.add_argument("--failed_indices", type=list, default=['demo_3', 'demo_6', 'demo_7'],
    #                     help="삭제할 실패 에피소드 인덱스 리스트")

    parser.add_argument("--failed_keys", nargs='+', 
                        default=['demo_14', 'demo_17', 'demo_18'],
                        help="삭제할 실패 에피소드 키 리스트 (예: demo_3 demo_6 demo_7)")

    args = parser.parse_args()

    # 기본 파일 경로 설정
    if args.converted_path is None:
        file_name, file_ext = os.path.splitext(args.hdf5_path)
        args.converted_path = f"{file_name}_replay{file_ext}"
    
    if args.inspected_file_path is None:
        file_name, file_ext = os.path.splitext(args.converted_path)
        args.inspected_file_path = f"{file_name}_inspected{file_ext}"
        
    if args.final_path is None:
        file_name, file_ext = os.path.splitext(args.inspected_file_path)
        args.final_path = f"{file_name}_final{file_ext}"

    # HDF5 재생용 변환
    # try:
    #     print(f"원본 파일을 재생용으로 변환 중: {args.hdf5_path} -> {args.converted_path}")
    #     hdf5_for_replay(args.hdf5_path, args.converted_path)
    # except Exception as e:
    #     print(f"변환 중 오류 발생: {e}")
    #     import traceback
    #     traceback.print_exc()

    # print(f"실패한 에피소드 삭제 및 재정렬 중: {args.converted_path} -> {args.inspected_file_path}")
    # delete_failed_episodes(args.converted_path, args.inspected_file_path, args.failed_keys)

    print(f"추가 데이터 병합 중: {args.inspected_file_path} + {args.supplement_hdf5_path} -> {args.final_path}")
    combine_episodes(args.inspected_file_path, args.supplement_hdf5_path, args.final_path)
    
    print("모든 처리가 완료되었습니다!")