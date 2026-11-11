"""Буфер опыта с равномерной выборкой и отдельным генератором случайных чисел."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.obs = np.empty((capacity, obs_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.terminated = np.empty(capacity, np.float32)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if batch_size < 1 or self.size < batch_size:
            raise ValueError("Not enough transitions for a minibatch")
        ix = self.rng.choice(self.size, batch_size, replace=False)
        return tuple(torch.from_numpy(a[ix]) for a in
                     (self.obs, self.actions, self.rewards, self.next_obs, self.terminated))
