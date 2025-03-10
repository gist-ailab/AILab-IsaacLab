from typing import Dict
import torch
import numpy as np
import copy

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class IsaacLabDataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 task_name=None,
                ):
        """
        IsaacLab 데이터셋 클래스
        
        Args:
            zarr_path: zarr 데이터셋 경로
            horizon: 시퀀스 길이
            pad_before: 시작부분 패딩 크기
            pad_after: 끝부분 패딩 크기
            seed: 랜덤 시드
            val_ratio: 검증 데이터셋 비율
            max_train_episodes: 최대 훈련 에피소드 수
            task_name: 태스크 이름
        """
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        print(f"IsaacLabDataset initialized with {len(self.sampler)} samples from {zarr_path}")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # 필요한 경우 특정 필드를 정규화 제외할 수 있음
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        # normalizer['imagin_robot'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer
    
    def __len__(self) -> int:
        """
        데이터셋의 길이를 반환합니다.
        """
        return len(self.sampler)
    
    def _sample_to_data(self, sample):
        """
        샘플을 데이터 딕셔너리로 변환
        """
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

        data_dict = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos      # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data_dict
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        인덱스를 통해 데이터를 반환합니다.
        """
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data