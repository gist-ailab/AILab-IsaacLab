# import gym    # BSH DP3 원본 task에서 사용
import gymnasium as gym
from gymnasium import spaces
from typing import Any, TypeVar
import numpy as np
from termcolor import cprint

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")

class SimpleVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            steps_per_render=1,
        ):
        """
        When file_path is None, don't record.
        """
        
        self.env = env
        self._action_space: spaces.Space[WrapperActType] | None = None
        self._observation_space: spaces.Space[WrapperObsType] | None = None
        self._metadata: dict[str, Any] | None = None
        self._reward_range = None

        self.mode = mode
        self.steps_per_render = steps_per_render

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()

        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        self.step_count = 1
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        
        frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result
    
    def get_video(self):
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        return video

