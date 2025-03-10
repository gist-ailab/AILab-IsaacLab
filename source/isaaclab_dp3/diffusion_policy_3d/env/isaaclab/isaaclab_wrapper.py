import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Tuple

# Isaac Lab 임포트
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.cam_ik_rel_env_cfg import FrankaCubeLiftEnvCfg


class IsaacLabEnv(gym.Env):
    """Isaac Lab 환경을 위한 래퍼 클래스
    
    이 클래스는 ManagerBasedRLEnv를 활용하여 Diffusion Policy에서 사용할 수 있는 인터페이스를 제공합니다.
    metaworld, adroit, dexart 래퍼들과 유사한 API를 제공합니다.
    """
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}
    
    def __init__(self, env_config: Dict[str, Any]):
        """
        Args:
            env_config: 환경 설정을 포함하는 딕셔너리. 다음 키를 포함해야 함:
                - task_name: 환경의 이름 (예: "Isaac-Lift-v0")
                - num_envs: 환경 인스턴스 수
                - headless: 시각화 여부
                - device: 시뮬레이션 디바이스 (예: "cuda:0")
                - max_steps: 최대 스텝 수
        """
        # 환경 설정 저장
        self.env_config = env_config
        self.device = env_config.get("device", "cuda:0")
        self.headless = env_config.get("headless", True)
        self.max_steps = env_config.get("max_steps", 1000)
        
        # 환경 설정 파싱
        task_name = env_config["task_name"]
        # Isaac Lab에서 제공하는 환경 설정 파싱 유틸리티 사용
        env_cfg = parse_env_cfg(
            task_name, 
            device=self.device,
            num_envs=env_config.get("num_envs", 1)
        )
        
        # 추가 설정 적용
        for key, value in env_config.items():
            if key not in ["task_name"]:
                setattr(env_cfg, key, value)
        
        # # 렌더링 모드 설정
        # render_mode = "rgb_array" if self.headless else "human"

        # # create environment configuration
        # env_cfg = FrankaCubeLiftEnvCfg()
        # env_cfg.scene.num_envs = 1
        
        # ManagerBasedRLEnv 환경 생성 (직접 인스턴스화)
        self.env = ManagerBasedRLEnv(cfg=env_cfg)
        
        # 액션 및 관측 공간 가져오기
        self.action_space = self.env.action_space
        self.observation_space_raw = self.env.observation_space
        
        # Diffusion Policy에서 사용할 관측 공간 정의
        self.observation_space = gym.spaces.Dict({
            # 정책용 관측 공간
            'policy': gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=self.observation_space_raw['policy'].shape, 
                dtype=np.float32
            ),
            # 에이전트 위치 정보
            'agent_pos': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self.observation_space_raw['policy'].shape,
                dtype=np.float32
            )
        })
        
        # 포인트 클라우드가 있는 경우 추가
        if 'point_cloud' in self.observation_space_raw:
            self.observation_space['point_cloud'] = self.observation_space_raw['point_cloud']
        
        # 이미지가 있는 경우 추가
        if 'rgb' in self.observation_space_raw:
            self.observation_space['image'] = self.observation_space_raw['rgb']
        
        # 카운터 및 에피소드 정보 초기화
        self.current_step = 0
        self.episode_reward = 0.0
        
    def reset(self):
        """환경을 초기화합니다."""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # ManagerBasedRLEnv의 reset 메소드는 (obs, info)를 반환
        obs, info = self.env.reset()
        
        # Diffusion Policy 형식으로 변환된 관측 반환
        return self._convert_to_obs_dict(obs), info
        
    def step(self, action):
        """액션을 수행하고 결과를 반환합니다.
        
        Args:
            action: 환경에 적용할 액션
            
        Returns:
            tuple: (obs_dict, reward, terminated, truncated, info)
                - obs_dict: 관측 딕셔너리
                - reward: 보상
                - terminated: 에피소드 종료 여부 (실패 또는 성공)
                - truncated: 에피소드 중단 여부 (시간 제한)
                - info: 추가 정보
        """
        # ManagerBasedRLEnv의 step 메소드는 (obs, reward, terminated, truncated, info)를 반환
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 스텝 카운터 증가 및 누적 보상 업데이트
        self.current_step += 1
        self.episode_reward += reward
        
        # 최대 스텝 수에 도달하면 종료
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Diffusion Policy 형식으로 변환된 관측 반환
        return self._convert_to_obs_dict(obs), reward, terminated, truncated, info
        
    def _convert_to_obs_dict(self, raw_obs):
        """원시 관측을 Diffusion Policy 형식으로 변환합니다."""
        # 기본 관측 딕셔너리 생성
        obs_dict = {
            'policy': raw_obs['policy'],
            'agent_pos': raw_obs['policy']  # 기본적으로 policy를 agent_pos로도 사용
        }
        
        # 추가 관측 키가 있으면 처리
        if 'point_cloud' in raw_obs:
            obs_dict['point_cloud'] = raw_obs['point_cloud']
        
        if 'rgb' in raw_obs:
            obs_dict['image'] = raw_obs['rgb']
        
        if 'depth' in raw_obs:
            obs_dict['depth'] = raw_obs['depth']
                
        return obs_dict
        
    def close(self):
        """환경을 종료합니다."""
        if self.env is not None:
            self.env.close()
            
    def render(self, mode='rgb_array'):
        """환경을 렌더링합니다."""
        # TODO: Isaac Lab 환경의 렌더링 메서드 사용할 것
        '''
        def render_high_res(self, resolution=1024):
            img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
            return img
        
        '''
        if mode == 'rgb_array':
            if hasattr(self.env, 'render'):
                return self.env.render(mode='rgb_array')
            else:
                # 렌더링이 지원되지 않는 경우 빈 프레임 반환
                return np.zeros((3, 84, 84), dtype=np.uint8)
        else:
            raise NotImplementedError(f"렌더링 모드 '{mode}'가 지원되지 않습니다.")
            
    def get_video(self):
        """비디오 프레임 수집"""
        if hasattr(self.env, "_rgb_annotator"):
            return self.env.render()
        return None
            
    def seed(self, seed=None):
        """난수 시드를 설정합니다."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return None
    
    @property
    def num_steps(self):
        """현재까지의 환경 스텝 수를 반환합니다."""
        if hasattr(self.env, 'common_step_counter'):
            return self.env.common_step_counter
        return self.current_step
        
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