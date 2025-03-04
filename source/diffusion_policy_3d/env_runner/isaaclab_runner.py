import os
import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply

# 올바른 임포트 경로
# from isaaclab_dp3.diffusion_policy_3d.env.isaaclab.isaaclab_wrapper import IsaacLabWrapper
# from ..env.isaaclab.isaaclab_wrapper import IsaacLabWrapper
from diffusion_policy_3d.env.isaaclab.isaaclab_wrapper import IsaacLabWrapper

class IsaacLabRunner(BaseRunner):
    """Isaac Lab 환경에서 정책을 평가하기 위한 러너 클래스"""
    
    def __init__(self, output_dir, env_config, eval_episodes=20, max_steps=1000, tqdm_interval_sec=5.0, device="cuda:0"):
        """
        Args:
            output_dir: 출력 디렉토리 경로
            env_config: 환경 설정 딕셔너리
            eval_episodes: 평가할 에피소드 수
            max_steps: 각 에피소드의 최대 스텝 수
            tqdm_interval_sec: tqdm 업데이트 간격(초)
            device: 사용할 디바이스
        """
        super().__init__(output_dir)
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device
        self.env_config = env_config

    def run(self, policy: BasePolicy, dataset=None, save_video=False):
        """
        주어진 정책을 사용하여 Isaac Lab 환경에서 평가를 실행합니다.
        
        Args:
            policy: 평가할 정책
            dataset: (선택 사항) 데이터셋 (사용하지 않음)
            save_video: 비디오를 저장할지 여부
        
        Returns:
            평가 결과가 포함된 딕셔너리
        """
        device = policy.device
        dtype = policy.dtype
        
        # 환경 생성
        env = IsaacLabWrapper(self.env_config)
        
        video_frames = []
        all_traj_rewards = []
        all_success_rates = []
        episode_lengths = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), 
                                    desc="Evaluating in Isaac Lab", 
                                    leave=False, 
                                    mininterval=self.tqdm_interval_sec):
            # 에피소드 시작
            obs = env.reset()
            policy.reset()
            
            episode_frames = []
            traj_reward = 0
            is_success = False
            
            # 에피소드 실행
            for step_idx in range(self.max_steps):
                # 관측을 정책 입력 형식으로 변환
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device, dtype=dtype))
                
                # 정책으로부터 액션 예측
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                
                # 액션을 numpy 배열로 변환
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
                action = np_action_dict['action']
                
                # 환경에서 한 스텝 실행
                obs, reward, done, info = env.step(action)
                traj_reward += reward
                
                # 성공 여부 확인
                if 'success' in info:
                    is_success = is_success or info['success']
                
                # 비디오 프레임 저장 (옵션)
                if save_video:
                    frame = env.render(mode='rgb_array')
                    if frame is not None:
                        episode_frames.append(frame)
                
                if done:
                    break
            
            # 에피소드 결과 기록
            all_traj_rewards.append(traj_reward)
            all_success_rates.append(float(is_success))
            episode_lengths.append(step_idx + 1)
            
            # 비디오 프레임 저장
            if save_video and len(episode_frames) > 0:
                video_frames.append(np.stack(episode_frames))
                
        # 환경 종료
        env.close()
        
        # 결과 통계 계산
        mean_traj_reward = np.mean(all_traj_rewards)
        mean_success_rate = np.mean(all_success_rates)
        mean_episode_length = np.mean(episode_lengths)
        
        # 로그 데이터 구성
        log_data = {
            'test_mean_reward': mean_traj_reward,
            'test_mean_success_rate': mean_success_rate,
            'test_mean_episode_length': mean_episode_length,
            'test_mean_score': mean_success_rate  # 성공률을 주요 평가 지표로 사용
        }
        
        # 비디오 로깅 (설정된 경우)
        if save_video and len(video_frames) > 0:
            # wandb에 비디오 로깅
            video_path = os.path.join(self.output_dir, f"eval_video_ep{self.eval_episodes}.mp4")
            
            # 여기에 비디오 저장 로직 추가
            # (예: imageio 또는 opencv를 사용하여 저장)
            
            if wandb.run is not None:
                wandb.log({
                    "eval_video": wandb.Video(video_path, fps=30, format="mp4")
                })
        
        return log_data