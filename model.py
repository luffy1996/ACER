# -*- coding: utf-8 -*-
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from time import sleep
import math

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    self.state_size = observation_space.shape[0]
    self.action_size = action_space.n

    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, self.action_size)

  def forward(self, x, h):
    x = F.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h


class ContinousActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size, sigma=0.3):
    super(ContinousActorCritic, self).__init__()
    self.sigma = torch.tensor([sigma])
    self.state_size = observation_space.shape[0]
    # Action Space for COntinous Space
    self.action_size = action_space.shape[0]

    self.relu = nn.ReLU(inplace=True)
    
    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)

    self.fc_actor = nn.Linear(hidden_size, self.action_size)

    self.fc_critic_value = nn.Linear(hidden_size, 1)

    self.action_input_layer = nn.Linear(self.action_size, hidden_size)

    self.fc_critic_advantage = nn.Linear(hidden_size, 1)

  def forward(self, x0):
    Q = None
    state = x0
    x1 = self.relu(self.fc1(x0))
    h = self.fc2(x1)  # h is (hidden state, cell state)
    x = h # TODO : Remove this line from code
    policy = self.fc_actor(x)  # Prevent 1s and hence NaNs
    V = self.fc_critic_value(x)

    # action = policy.data + torch.normal(torch.zeros(policy.size()), torch.ones(policy.size())*0.01)
    action = (policy + self.sigma.sqrt()*torch.randn( policy.size() )).data
    action_samples = [Variable( (policy + self.sigma.sqrt()*torch.randn( policy.size() ).data) ) for _ in range(5)]
    advantage_samples = torch.cat([self.Advantage(x, action_sample).unsqueeze(-1) for action_sample in action_samples], -1)

    A = self.Advantage(x, action)
    Q = V + A - advantage_samples.mean(-1)
    return policy, Q, V, action

  
  def Advantage(self, x, action):
    hidden = x + self.action_input_layer(action)
    A = self.fc_critic_advantage(hidden)
    return A

  