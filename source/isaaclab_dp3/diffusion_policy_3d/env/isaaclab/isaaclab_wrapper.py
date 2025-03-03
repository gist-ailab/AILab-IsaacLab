import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List

# 올바른 Isaac Lab 임포트
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class IsaacLabWrapper:
    """Isaac Lab 환경을 위한 래퍼 클래스
    
    이 클래스는 Isaac Lab 환경을 감싸고 Diffusion Policy에서 사용할 수 있도록 인터페이스를 제공합니다.
    """
    def __init__(self, env_config: Dict[str, Any]):
        """
        Args:
            env_config: 환경 설정을 포함하는 딕셔너리. 다음 키를 포함해야 함:
                - task_name: 환경의 이름 (예: "Isaac-Cartpole-v0")
                - num_envs: 환경 인스턴스 수
                - headless: 시각화 여부
                - device: 시뮬레이션 디바이스 (예: "cuda:0")
        """
        # 환경 설정 저장
        self.env_config = env_config
        
        # 환경 설정 파싱
        task_name = env_config["task_name"]
        # Isaac Lab에서 제공하는 환경 설정 파싱 유틸리티 사용
        env_cfg = parse_env_cfg(
            task_name, 
            device=env_config.get("device", "cuda:0"),
            num_envs=env_config.get("num_envs", 1)
        )
        
        # 추가 설정 적용
        for key, value in env_config.items():
            if key not in ["task_name"]:
                setattr(env_cfg, key, value)
        
        # 환경 생성
        self.env = gym.make(task_name, cfg=env_cfg)
        
        # 액션 및 관측 공간 가져오기
        self.action_space = self.env.action_space
        self.observation_space_raw = self.env.observation_space
        
        # Diffusion Policy에서 사용할 관측 공간 정의
        # 참고: 실제 환경의 관측 공간에 맞게 수정 필요
        self.observation_space = gym.spaces.Dict({
            # 'policy' 키는 Diffusion Policy에서 사용할 정책 입력
            'policy': gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=self.observation_space_raw['policy'].shape, 
                dtype=np.float32
            )
        })
        
        # 렌더링 관련 변수
        self.frames = []
        
    def reset(self):
        """환경을 초기화합니다."""
        obs, _ = self.env.reset()
        return self._convert_to_obs_dict(obs)
        
    def step(self, action):
        """액션을 수행하고 결과를 반환합니다."""
        next_obs, reward, done, info = self.env.step(action)
        return self._convert_to_obs_dict(next_obs), reward, done, info
        
    def _convert_to_obs_dict(self, raw_obs):
        """원시 관측을 Diffusion Policy 형식으로 변환합니다."""
        # Isaac Lab 환경의 관측은 이미 Dict 형태로 제공
        # 여기서는 'policy' 키에 해당하는 부분만 사용
        obs_dict = {
            'policy': raw_obs['policy'],
        }
        
        # 필요한 경우 추가 관측 키를 처리
        # 예: 'image', 'depth', 'agent_pos', 'point_cloud' 등
        for key in ['image', 'depth', 'agent_pos', 'point_cloud']:
            if key in raw_obs:
                obs_dict[key] = raw_obs[key]
                
        return obs_dict
        
    def close(self):
        """환경을 종료합니다."""
        if self.env is not None:
            self.env.close()
            
    def render(self, mode='rgb_array'):
        """환경을 렌더링합니다."""
        if mode == 'rgb_array':
            if hasattr(self.env, 'render'):
                return self.env.render(mode='rgb_array')
            else:
                # 렌더링이 지원되지 않는 경우 빈 프레임 반환
                return np.zeros((3, 84, 84), dtype=np.uint8)
        else:
            raise NotImplementedError(f"렌더링 모드 '{mode}'가 지원되지 않습니다.")
            
    def seed(self, seed=None):
        """난수 시드를 설정합니다."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return None
        
    def get_state(self):
        """환경의 상태를 가져옵니다."""
        if hasattr(self.env, 'get_state'):
            return self.env.get_state()
        raise NotImplementedError("get_state 메서드가 지원되지 않습니다.")
        
    def set_state(self, state):
        """환경의 상태를 설정합니다."""
        if hasattr(self.env, 'set_state'):
            return self.env.set_state(state)
        raise NotImplementedError("set_state 메서드가 지원되지 않습니다.")