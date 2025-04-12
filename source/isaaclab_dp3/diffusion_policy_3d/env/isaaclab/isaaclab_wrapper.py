import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Tuple

# Isaac Lab 임포트
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.cam_ik_rel_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat

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
        self.device = env_config["device"]
        self.headless = env_config["headless"]
        self.max_steps = env_config["max_steps"]
        self.render_mode = env_config["render_mode"]
        
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
        
        # ManagerBasedRLEnv 환경 생성 (직접 인스턴스화)
        self.env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=self.render_mode)
        
        # 액션 및 관측 공간 가져오기
        # self.action_space = self.env.action_space
        self.action_space = gym.spaces.Box(
                low=self.env.action_space.low[0],
                high=self.env.action_space.high[0],
                shape=(self.env.action_space.shape[1],),
                dtype=self.env.action_space.dtype
        )
        self.observation_space_raw = self.env.observation_space
        
        # Diffusion Policy에서 사용할 관측 공간 정의
        self.observation_space = gym.spaces.Dict({
            'point_cloud': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self.observation_space_raw['vision']['point_cloud'].shape,
                dtype=np.float32
            ),
            # # rgb_image는 안 써서 주석처리
            # 'rgb_image': gym.spaces.Box(
            #     low=-np.inf, high=np.inf,
            #     shape=self.observation_space_raw['vision']['rgb_image'].shape,
            #     dtype=np.int32
            # ),
            'agent_pos': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self.observation_space_raw['policy']['agent_pos'].shape,
                dtype=np.float32
            ),
        })

        # 카운터 정보 초기화
        self.current_step = 0
        
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """환경을 초기화합니다."""
        self.current_step = 0
        
        # ManagerBasedRLEnv의 reset 메소드는 (obs, info)를 반환
        obs, info = self.env.reset()
        
        # Diffusion Policy 형식으로 변환된 관측 반환
        return self._convert_to_obs_dict(obs)
        
    def step(self, action):
        """액션을 수행하고 결과를 반환합니다.
        
        Args:
            action: 환경에 적용할 액션
            
        Returns:
            tuple: (obs_dict, reward, terminated, truncated, info)
                - obs_dict: 관측 딕셔너리
                - reward: 보상
                - terminated: 에피소드 종료 여부 (실패 또는 성공)
                - time_out: 에피소드 중단 여부 (시간 제한)
                - info: 추가 정보
        """
        # ManagerBasedRLEnv의 step 메소드는 (obs, reward, terminated, truncated, info)를 반환
        action_euler = torch.zeros_like(action)[..., :-1]   # euler는 quaternion보다 하나 작으므로 slicing 함.
        action_euler[:, 0:3] = action[:, 0:3]   # copy x, y, z position
        action_euler[:, -1] = action[:, -1]     # copy gripper action
        
        # quaternion을 euler_xyz로 변환
        (roll, pitch, yaw) = euler_xyz_from_quat(action[:, 3:7])
        euler_xyz = torch.stack((roll, pitch, yaw), dim=1)
        action_euler[:, 3:6] = euler_xyz
        
        # action이 두 frame 사이의 pose 변화량으로 학습했기 때문에 scale해주는 것이 필요함.
        action_euler = action_euler * 10
        obs, reward, terminated, time_out, info = self.env.step(action_euler)
        
        # 스텝 카운터 증가 및 누적 보상 업데이트
        self.current_step += 1
        
        # observation과 성공 여부(terminated)를 반환.
        # terminated는 tensor([False], device='cuda:0') 형태라 1번째 요소만 반환
        return self._convert_to_obs_dict(obs), terminated[0]
        
    def _convert_to_obs_dict(self, raw_obs):
        """원시 관측을 Diffusion Policy 형식으로 변환합니다."""
        # 기본 관측 딕셔너리 생성
        obs_dict = {
            'point_cloud': raw_obs['vision']['point_cloud'],
            # 'rgb_image': raw_obs['vision_robot']['rgb_image'].squeeze(),    # 차원 맞추기 위해 squeeze, 안 써서 주석처리
            'agent_pos': raw_obs['policy']['agent_pos'],
        }
                
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
                # return self.env.render(mode='rgb_array')
                return self.env.render()
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