import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy_3d.env.isaaclab.isaaclab_wrapper import IsaacLabEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

# 어댑터 클래스 임포트
from diffusion_policy_3d.env_runner.dp3_isaaclab_adapter import DP3IsaacLabAdapter

class IsaacLabRunner(BaseRunner):
    """Isaac Lab 환경에서 정책을 평가하기 위한 러너 클래스"""
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=2,
                 n_action_steps=3,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        """
        Args:
            output_dir: 출력 디렉토리 경로
            eval_episodes: 평가할 에피소드 수
            max_steps: 각 에피소드의 최대 스텝 수
            n_obs_steps: 관측 스텝 수
            n_action_steps: 액션 스텝 수
            fps: 프레임 레이트
            tqdm_interval_sec: tqdm 업데이트 간격(초)
            device: 실행 디바이스
            task_name: 태스크 이름
        """
        super().__init__(output_dir)
        self.task_name = task_name
           
        # 환경 생성 함수 (MultiStepWrapper 제거)
        def env_fn(task_name):
            return SimpleVideoRecordingWrapper(
                IsaacLabEnv({
                    "task_name": task_name,
                    "num_envs": 1,  # 단일 환경 사용
                    "headless": True,
                    "device": device,
                    "max_steps": max_steps,
                    "render_mode": "rgb_array",
                })
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)
        
        # 평가 설정
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device
        
        # 로거 설정
        self.logger_util = logger_util.LargestKRecorder(K=3)
        self.logger_util_top5 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, **kwargs):
        """
        정책을 실행하고 평가합니다.
        
        Args:
            policy: 평가할 정책
            
        Returns:
            평가 결과를 담은 딕셔너리
        """
        device = policy.device
        dtype = policy.dtype
        
        # 어댑터로 정책 래핑
        policy_adapter = DP3IsaacLabAdapter(
            policy, 
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps,
            device=device
        )

        all_rewards = torch.zeros(self.eval_episodes, device=device, dtype=dtype)
        all_success_rates = torch.zeros(self.eval_episodes, device=device, dtype=torch.bool)
        env = self.env
        
        ##############################
        # env loop
        for episode_idx in tqdm.tqdm(range(self.eval_episodes),
                                     desc=f"Isaac Lab {self.task_name} 환경에서 평가 중",
                                     leave=False,
                                     mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env.reset()
            # policy.reset()    # policy_adapter.reset()으로 변경
            policy_adapter.reset()
            
            done = False
            total_reward = 0
            episode_step = 0
            
            # 에피소드 실행
            while not done and episode_step < self.max_steps:
                # 어댑터를 통해 n_action_steps만큼 환경 진행
                obs, reward, done, info = policy_adapter.step(env, obs)
                
                total_reward += reward
                episode_step += policy_adapter.n_action_steps
                
                # # 디버깅 출력
                # if episode_step % 50 == 0:
                #     print(f"Episode {episode_idx}, Step {episode_step}, Reward: {reward}, Done: {done}")
                
                # 성공 여부 확인
                if 'success' in info:
                    success = info['success']
                else:
                    success = False
                    
                # 최대 스텝 수 체크
                if episode_step >= self.max_steps:
                    # print(f"Episode {episode_idx} reached max steps limit.")
                    done = True
            
            # 에피소드 결과 기록
            all_rewards[episode_idx] = total_reward.clone().detach().to(device=device, dtype=dtype)
            all_success_rates[episode_idx] = torch.tensor(success, device=device, dtype=torch.bool)
        
        # 평가 결과 정리
        log_data = {}
        mean_reward = torch.mean(all_rewards)
        mean_success_rate = torch.mean(all_success_rates.float()).item()
        
        log_data['mean_reward'] = mean_reward
        log_data['mean_success_rate'] = mean_success_rate
        log_data['test_mean_score'] = mean_success_rate
        
        # 상위 K개 결과 기록
        self.logger_util.record(mean_success_rate)
        self.logger_util_top5.record(mean_success_rate)
        log_data['success_rate_top3'] = self.logger_util.average_of_largest_K()
        log_data['success_rate_top5'] = self.logger_util_top5.average_of_largest_K()
        
        cprint(f"Test Mean Success Rate: {mean_success_rate:.4f}", 'green')
        
        return log_data