# dp3_isaaclab_adapter.py

import torch
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Union, List

from diffusion_policy_3d.policy.base_policy import BasePolicy


class DP3IsaacLabAdapter:
    """
    DP3와 Isaac Lab 환경 간의 인터페이스 어댑터 클래스
    
    이 클래스는 DP3 정책의 출력(batch_size, n_action_steps, action_dim)을
    Isaac Lab 환경이 기대하는 입력(num_envs, action_dim)으로 변환합니다.
    또한 Isaac Lab 환경의 observation을 DP3 정책이 기대하는 형태로 변환합니다.
    """
    
    def __init__(self, 
                 policy: BasePolicy, 
                 n_obs_steps: int = 2, 
                 n_action_steps: int = 3, 
                 max_episode_steps: int = 1000,
                 device: Optional[torch.device] = None):
        """
        Args:
            policy: DP3 정책 인스턴스
            n_obs_steps: 관측 스텝 수
            n_action_steps: 액션 스텝 수
            num_envs: 환경 인스턴스 수
            device: 계산 디바이스 (None인 경우 policy.device 사용)
        """
        self.policy = policy
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.device = device if device is not None else policy.device
        
        # 상태 초기화
        self.reset()
        
    def reset(self):
        """환경이 리셋될 때 호출"""
        # observation 히스토리 저장을 위한 버퍼 (각 키별로 관리)
        self.obs_buffer = {}
        
        # 에피소드 진행 중 수집된 데이터
        self.reward_buffer = []
        self.done_buffer = []
        self.info_buffer = defaultdict(list)
        
        # 현재 에피소드 스텝 카운터
        self.step_count = 0
        
        # 현재 action 인덱스
        self.current_action_idx = 0
        
        # 현재 action 시퀀스 (None이면 새로 예측 필요)
        self.current_actions = None
        
    def process_observation(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Isaac Lab 환경에서 받은 observation을 DP3 모델에 맞게 변환
        
        Args:
            obs_dict: Isaac Lab 환경에서 받은 observation 딕셔너리
            
        Returns:
            DP3 모델에 입력할 형태로 변환된 observation 딕셔너리
        """
        processed_obs = {}

        if self.step_count == 0:
            for key, value in obs_dict.items():
                processed_obs[key] = torch.stack((obs_dict[key], obs_dict[key]), dim=0)

        else:
            processed_obs = obs_dict

        # # 순서 뒤집기 (가장 오래된 observation이 먼저 오도록)
        # obs_list.reverse()
        # TODO: reverse 필요할까?? 어떻게 쌓이고 있는지, 다른 예제는 어떤 순서로 쌓고 있는 건지 확인 필요



        # 첫 번째 호출인 경우 obs_buffer 초기화
        if not self.obs_buffer:
            # 각 키별로 deque 초기화
            for key, value in obs_dict.items():
                self.obs_buffer[key] = deque(maxlen=self.n_obs_steps)
        
        # 현재 observation을 각 키별로 버퍼에 추가
        for key, value in obs_dict.items():
            # 관측값의 차원에 따라 일관되게 처리
            # 1차원 텐서인 경우 (예: agent_pos [9])
            if value.dim() == 1:
                # 1차원 텐서는 그대로 버퍼에 추가
                self.obs_buffer[key].append(value)
            # 2차원 이상 텐서인 경우 (예: point_cloud [1024, 6], rgb_image [576, 640, 3])
            elif value.dim() >= 2:
                # 이미 n_obs_steps 차원이 있는 경우 (step 메소드에서 반환된 경우)
                if value.dim() >= 3 and value.shape[0] == self.n_obs_steps:
                    # 가장 최근 observation만 사용 (마지막 스텝)
                    self.obs_buffer[key].append(value[-1])
                else:
                    # 일반적인 경우: 단일 observation을 버퍼에 추가
                    self.obs_buffer[key].append(value)
        
        # 버퍼에서 observation 히스토리 구성
        if all(len(buffer) > 0 for buffer in self.obs_buffer.values()):
            # 각 키별로 observation 스택
            for key, buffer in self.obs_buffer.items():
                # 관측값의 차원에 따라 다르게 처리
                sample_obs = buffer[0]  # 첫 번째 관측값으로 차원 확인
                
                if sample_obs.dim() == 1:
                    # 1차원 텐서인 경우 (예: agent_pos [9])
                    # 버퍼 내용 검사 및 정규화
                    normalized_buffer = []
                    for obs in buffer:
                        # 모든 텐서가 1차원인지 확인
                        if obs.dim() == 1:
                            normalized_buffer.append(obs)
                        elif obs.dim() > 1:
                            # 2차원 이상인 경우 마지막 차원만 사용
                            normalized_buffer.append(obs.reshape(-1))
                    
                    # 정규화된 버퍼로 교체
                    buffer_list = list(normalized_buffer)
                    
                    # 히스토리 구성
                    obs_list = []
                    for i in range(self.n_obs_steps):
                        if i < len(buffer_list):
                            obs_list.append(buffer_list[-(i+1)])
                        else:
                            obs_list.append(buffer_list[0])  # 패딩
                    
                    # 순서 뒤집기 (가장 오래된 observation이 먼저 오도록)
                    obs_list.reverse()
                    
                    # 모든 텐서가 동일한 크기인지 확인
                    first_shape = obs_list[0].shape
                    for i, obs in enumerate(obs_list):
                        if obs.shape != first_shape:
                            print(f"Warning: Inconsistent shapes in obs_list at index {i}. Expected {first_shape}, got {obs.shape}")
                            # 크기가 다른 경우 첫 번째 텐서와 동일한 크기로 조정
                            obs_list[i] = obs_list[0].clone()
                    
                    # 직접 스택하여 [n_obs_steps, feature_dim] 형태로 만듦
                    stacked_obs = torch.stack(obs_list)  # [n_obs_steps, feature_dim]
                
                elif sample_obs.dim() >= 2:
                    # 2차원 이상 텐서인 경우 (예: point_cloud [1024, 6], rgb_image [576, 640, 3])
                    # 버퍼 내용 검사 및 정규화
                    normalized_buffer = []
                    first_obs = buffer[0]
                    first_shape = first_obs.shape
                    
                    for obs in buffer:
                        if obs.shape == first_shape:
                            normalized_buffer.append(obs)
                        else:
                            # 차원이 다른 경우 첫 번째 텐서와 동일한 형태로 조정
                            print(f"Warning: Inconsistent shapes in buffer. Expected {first_shape}, got {obs.shape}")
                            normalized_buffer.append(first_obs.clone())
                    
                    # 정규화된 버퍼로 교체
                    buffer_list = list(normalized_buffer)
                    
                    # 히스토리 구성
                    obs_list = []
                    for i in range(self.n_obs_steps):
                        if i < len(buffer_list):
                            obs_list.append(buffer_list[-(i+1)])
                        else:
                            obs_list.append(buffer_list[0])  # 패딩
                    
                    # 순서 뒤집기 (가장 오래된 observation이 먼저 오도록)
                    obs_list.reverse()
                    
                    # 모든 텐서가 동일한 크기인지 한 번 더 확인
                    for i, obs in enumerate(obs_list):
                        if obs.shape != first_shape:
                            print(f"Warning: Inconsistent shapes in obs_list at index {i}. Expected {first_shape}, got {obs.shape}")
                            # 크기가 다른 경우 첫 번째 텐서와 동일한 크기로 조정
                            obs_list[i] = obs_list[0].clone()
                    
                    # 스택하여 [n_obs_steps, ...] 형태로 만듦
                    stacked_obs = torch.stack(obs_list)
                
                processed_obs[key] = stacked_obs.unsqueeze(0)  # 배치 차원 추가 [1, 2, ...]
        
        return processed_obs
        
    # region
    # def predict_action(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
    #     """
    #     DP3 모델을 사용하여 action 예측
        
    #     Args:
    #         obs_dict: Isaac Lab 환경에서 받은 observation 딕셔너리
                      
    #     Returns:
    #         action: Isaac Lab 환경에 입력할 action
    #                형태: (num_envs, action_dim)
    #     """
    #     # observation 처리
    #     dp3_obs_dict = self.process_observation(obs_dict)
        
    #     # 새로운 action sequence가 필요한 경우에만 DP3 모델 호출
    #     if self.current_step_idx == 0 or self.action_buffer is None:
    #         with torch.no_grad():
    #             action_dict = self.policy.predict_action(dp3_obs_dict)
    #             # action 형태: (batch_size, n_action_steps, action_dim)
    #             self.current_actions = action_dict['action']
    #             self.current_action_idx = 0
        
    #     # 현재 step에 해당하는 action 선택
    #     current_action = self.current_actions[:, self.current_action_idx]
        
    #     # 다음 step으로 이동
    #     self.current_action_idx = (self.current_action_idx + 1) % self.n_action_steps
        
    #     # 스텝 카운터 증가
    #     self.step_count += 1
        
    #     # Isaac Lab 형식에 맞게 변환: (1, action_dim)
    #     return current_action.reshape(1, -1)
    # endregion

    def step(self, env, obs):
        """
        n_action_steps만큼 환경을 진행하고 결과 반환
        
        Args:
            env: Isaac Lab 환경
            obs: 현재 observation
            
        Returns:
            tuple: (next_obs, reward, done, info)
                - next_obs: 다음 observation
                - reward: 누적 보상
                - done: 종료 여부
                - info: 추가 정보
        """
        # 버퍼 초기화
        self.reward_buffer = []
        self.done_buffer = []
        self.info_buffer = defaultdict(list)
        
        # 현재 observation을 버퍼에 추가
        obs_dict = dict(obs)
        
        # DP3 모델에서 action 시퀀스 예측
        with torch.no_grad():
            dp3_obs_dict = self.process_observation(obs_dict)
            action_dict = self.policy.predict_action(dp3_obs_dict)
            actions = action_dict['action']  # (batch_size, n_action_steps, action_dim)

        # 배치 차원 제거하고 각 action을 순차적으로 처리
        # (batch_size, n_action_steps, action_dim) -> (n_action_steps, action_dim)
        action_sequence = actions[0]
        
        # 각 action을 순차적으로 처리
        for act in action_sequence:
            # 이미 종료된 경우 중단
            if len(self.done_buffer) > 0 and self.done_buffer[-1]:
                break
                
            # 현재 action을 환경 입력 형태로 변환 (1, action_dim)
            current_action = act.reshape(1, -1)
            
            # 환경에 action 적용
            next_obs, reward, done, info = env.step(current_action)
            
            # 결과 저장
            self.reward_buffer.append(reward)
            
            # done 처리
            done = done.item()
                
            # 최대 에피소드 길이 체크 (truncation)
            if (self.max_episode_steps is not None) and (self.step_count >= self.max_episode_steps):
                done = True
            
            self.done_buffer.append(done)
            
            # observation 업데이트
            obs = next_obs
            
            # 각 키별로 observation 버퍼 업데이트
            obs_dict = dict(obs)
            for key, value in obs_dict.items():
                if key not in self.obs_buffer:
                    self.obs_buffer[key] = deque(maxlen=self.n_obs_steps)
                
                # 관측값의 차원에 따라 일관되게 처리
                # 1차원 텐서인 경우 (예: agent_pos [9])
                if value.dim() == 1:
                    # 1차원 텐서는 그대로 버퍼에 추가
                    self.obs_buffer[key].append(value)
                # 2차원 이상 텐서인 경우 (예: point_cloud [1024, 6], rgb_image [576, 640, 3])
                elif value.dim() >= 2:
                    # 이미 n_obs_steps 차원이 있는 경우 (step 메소드에서 반환된 경우)
                    if value.dim() >= 3 and value.shape[0] == self.n_obs_steps:
                        # 가장 최근 observation만 사용 (마지막 스텝)
                        self.obs_buffer[key].append(value[-1])
                    else:
                        # 일반적인 경우: 단일 observation을 버퍼에 추가
                        self.obs_buffer[key].append(value)
            
            # 스텝 카운터 증가
            self.step_count += 1
        
        # 결과 집계
        # 보상 집계 (sum)
        if len(self.reward_buffer) > 0:
            total_reward = torch.sum(torch.stack(self.reward_buffer)) if isinstance(self.reward_buffer[0], torch.Tensor) else sum(self.reward_buffer)
        else:
            total_reward = 0
            
        # 종료 여부 (하나라도 True면 True)
        done = any(self.done_buffer) if self.done_buffer else False
        
        # info 처리
        info = {}
        for key, values in self.info_buffer.items():
            if len(values) > 0:
                info[key] = values[-1]
        
        # 최종 observation 반환
        final_obs = self._get_obs(self.n_obs_steps)
        
        return final_obs, total_reward, done, info


    def _get_obs(self, n_obs_steps):
        """
        최근 n_obs_steps만큼의 observation을 반환
        
        Args:
            n_obs_steps: 반환할 observation 스텝 수
            
        Returns:
            dict: 최근 n_obs_steps만큼의 observation
        """
        if len(self.obs_buffer) == 0:
            return {}
            
        result = {}
        for key, buffer in self.obs_buffer.items():
            if len(buffer) == 0:
                continue
                
            # 관측값의 차원에 따라 다르게 처리
            sample_obs = buffer[0]  # 첫 번째 관측값으로 차원 확인
            
            if sample_obs.dim() == 1:
                # 1차원 텐서인 경우 (예: agent_pos [9])
                # 히스토리를 직접 스택하여 [n_obs_steps, feature_dim] 형태로 만듦
                obs_list = []
                for i in range(n_obs_steps):
                    if i < len(buffer):
                        obs_list.append(buffer[-(i+1)])
                    else:
                        obs_list.append(buffer[0])  # 패딩
                
                # 순서 뒤집기 (가장 오래된 observation이 먼저 오도록)
                obs_list.reverse()
                
                # 직접 스택하여 [n_obs_steps, feature_dim] 형태로 만듦
                stacked_obs = torch.stack(obs_list)  # [n_obs_steps, feature_dim]
            
            elif sample_obs.dim() >= 2:
                # 2차원 이상 텐서인 경우 (예: point_cloud [1024, 6], rgb_image [576, 640, 3])
                # 각 관측값을 그대로 유지하면서 스택
                obs_list = []
                for i in range(n_obs_steps):
                    if i < len(buffer):
                        obs_list.append(buffer[-(i+1)])
                    else:
                        obs_list.append(buffer[0])  # 패딩
                
                # 순서 뒤집기 (가장 오래된 observation이 먼저 오도록)
                obs_list.reverse()
                
                # 스택하여 [n_obs_steps, ...] 형태로 만듦
                try:
                    stacked_obs = torch.stack(obs_list)
                except RuntimeError as e:
                    # 차원이 일치하지 않는 경우 디버그 정보 출력
                    dims = [obs.shape for obs in obs_list]
                    print(f"Error stacking tensors with shapes: {dims}")
                    
                    # 차원을 일치시키기 위해 모든 텐서를 첫 번째 텐서와 동일한 차원으로 변환
                    first_shape = obs_list[0].shape
                    normalized_list = []
                    
                    for obs in obs_list:
                        if obs.shape != first_shape:
                            # 차원이 다른 경우, 첫 번째 텐서와 동일한 차원으로 변환
                            # 1차원 텐서를 2차원으로 변환하는 경우
                            if obs.dim() == 1 and len(first_shape) > 1:
                                normalized_list.append(obs.unsqueeze(0))
                            else:
                                # 안전하게 첫 번째 텐서로 대체
                                normalized_list.append(obs_list[0])
                        else:
                            normalized_list.append(obs)
                    
                    stacked_obs = torch.stack(normalized_list)
            
            result[key] = stacked_obs
        
        return result