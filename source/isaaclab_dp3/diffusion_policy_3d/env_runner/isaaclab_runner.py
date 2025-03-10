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

class IsaacLabRunner(BaseRunner):
    """Isaac Lab 환경에서 정책을 평가하기 위한 러너 클래스"""
    # def __init__(self,
    #              output_dir,
    #              eval_episodes=20,
    #              max_steps=30,
    #              n_obs_steps=8,
    #              n_action_steps=8,
    #              fps=10,
    #              tqdm_interval_sec=5.0,
    #              device="cuda:0",
    #              task_name="lift"):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
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
        

        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    IsaacLabEnv({
                        "task_name": task_name,
                        "num_envs": 1, 
                        "headless": True,
                        "device": device,
                        "max_steps": max_steps
                    })
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        # # 환경 설정
        # env_config = {
        #     "task_name": task_name,
        #     "num_envs": 1,  # 단일 환경 사용
        #     "headless": True,  # 헤드리스 모드
        #     "device": device,
        #     "max_steps": max_steps
        # }
        
        # # 환경 초기화
        # self.env = IsaacLabEnv(env_config)
        
        # 평가 설정
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        
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
        
        all_rewards = []
        all_success_rates = []
        env = self.env
        
        ##############################
        # train env loop
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), 
                                    desc=f"Isaac Lab {self.task_name} 환경에서 평가 중", 
                                    leave=False, 
                                    mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env.reset()
            policy.reset()
            
            done = False
            total_reward = 0
            
            # 에피소드 실행
            while not done:
                # 관측값을 정책 입력 형식으로 변환
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                     lambda x: torch.from_numpy(x).to(device=device) 
                                     if isinstance(x, np.ndarray) else x)
                
                # 배치 차원 추가
                with torch.no_grad():
                    obs_dict_input = {}
                    if 'point_cloud' in obs_dict:
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    if 'agent_pos' in obs_dict:
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    # 정책으로 액션 예측
                    action_dict = policy.predict_action(obs_dict_input)
                
                # 액션을 numpy 배열로 변환
                np_action_dict = dict_apply(action_dict,
                                          lambda x: x.detach().cpu().numpy())
                action = np_action_dict['action'].squeeze(0)
                
                # 환경에 액션 적용
                obs, reward, done, info = env.step(action)
                
                total_reward += reward
                # 성공 여부 확인 (info의 형태에 따라 달라질 수 있음)
                if 'success' in info:
                    success = info['success']
                else:
                    success = False
            
            # 에피소드 결과 기록
            all_rewards.append(total_reward)
            all_success_rates.append(success)
        
        # 평가 결과 정리
        log_data = {}
        mean_reward = np.mean(all_rewards)
        mean_success_rate = np.mean(all_success_rates)
        
        log_data['mean_reward'] = mean_reward
        log_data['mean_success_rate'] = mean_success_rate
        log_data['test_mean_score'] = mean_success_rate
        
        # 상위 K개 결과 기록
        self.logger_util.record(mean_success_rate)
        self.logger_util_top5.record(mean_success_rate)
        log_data['success_rate_top3'] = self.logger_util.average_of_largest_K()
        log_data['success_rate_top5'] = self.logger_util_top5.average_of_largest_K()
        
        cprint(f"Test Mean Success Rate: {mean_success_rate:.4f}", 'green')
        
        # 환경 초기화
        env.reset()
        
        return log_data