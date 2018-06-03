# -*- coding: utf-8 -*-
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from torch.autograd import Variable


def _multivariate_normal_pdf(x, mu, sigma=None):
  # Note that sigma is a 0.3 * diagnol matrix
  X = x - mu
  d = x.shape[-1]
  X = X**2
  X = X.sum()
  X = X*0.3
  f = torch.exp(-X)/( ((2*math.pi)**(1/d)) * 0.3)
  return f

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    self.state_size = observation_space.shape[0]
    self.action_size = action_space.n

    self.relu = nn.ReLU(inplace=True)
    self.softmax = nn.Softmax(dim=1)

    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, self.action_size)

  def forward(self, x, h):
    x = self.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h

class ContinousActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ContinousActorCritic, self).__init__()
    self.state_size = observation_space.shape[0]
    # Action Space for COntinous Space
    self.action_size = action_space.shape[0]

    self.relu = nn.ReLU(inplace=True)
    
    self.fc1 = nn.Linear(self.state_size, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    # The value is action independent
    self.fc_critic_value = nn.Linear(hidden_size, 1)

    self.action_input_layer = nn.Linear(self.action_size, hidden_size)

    self.fc_critic_advantage = nn.Linear(hidden_size, 1)

  def forward(self, x, h):
    Q = None
    x = self.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.fc_actor(x)  # Prevent 1s and hence NaNs
    V = self.fc_critic_value(x)
    # Adding state and action for stochiastic duelling network
    # TODO Add Noise
    actions = policy.data
    action_samples = [Variable(torch.normal(policy.data, torch.exp(torch.ones(policy.size(0), 1)))) for _ in range(5)]
    advantage_samples = torch.cat([self.Advantage(x, action_sample).unsqueeze(-1) for action_sample in action_samples], -1)
    A = self.Advantage(x, actions)
    Q = V + A - advantage_samples.mean(-1)
    return policy, Q, V, h, actions

  
  def Advantage(self, x, actions):
    hidden = x + self.action_input_layer(actions)
    A = self.fc_critic_advantage(hidden)
    return A

  