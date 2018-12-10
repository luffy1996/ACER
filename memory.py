# -*- coding: utf-8 -*-
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy', 'done'))


class EpisodicReplayMemory():
  def __init__(self, capacity, max_episode_length):
    # Max number of transitions possible will be the memory capacity, could be much less
    self.num_episodes = capacity // max_episode_length
    self.memory = deque(maxlen=self.num_episodes)
    self.trajectory = []
    self.mem_length = 0
    self.max_episode_length = max_episode_length

  def append(self, state, action, reward, policy, done):
    self.mem_length = (self.mem_length + 1) % self.max_episode_length
    self.trajectory.append(Transition(state, action, reward, policy, done))  # Save s_i, a_i, r_i+1, µ(·|s_i)
    # Terminal states are saved with actions as None, so switch to next episode
    if self.mem_length==0:
      # print ('done')
      self.memory.append(self.trajectory)
      self.trajectory = []
  # Samples random trajectory
  def sample(self, maxlen=0):
    mem = self.memory[random.randrange(len(self.memory))]
    # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
    return mem

  # Samples batch of trajectories, truncating them to the same length
  def sample_batch(self, batch_size, maxlen=0):
    batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
    return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

  def length(self):
    # Return number of epsiodes saved in memory
    return len(self.memory)

  def __len__(self):
    return sum(len(episode) for episode in self.memory)
