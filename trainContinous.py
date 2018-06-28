# -*- coding: utf-8 -*-
import math
import random
import gym
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
# from scipy.stats import multivariate_normal
from memory import EpisodicReplayMemory
from model import ActorCritic, ContinousActorCritic
from utils import state_to_tensor, plot_loss
import math
from time import sleep
import numpy as np
def _multivariate_normal_pdf(x, mu, sigma=None):
  # Assert that the shape of mu and x is same
  if not (x.shape == mu.shape):
    raise AssertionError()
  X = x - mu
  d = x.shape[-1]
  X = X**2
  X = X.sum()
  X = X/(0.09)
  f = torch.exp(-X*0.5)/ torch.sqrt( torch.tensor([2*math.pi*(0.09**(d))])) 
  return f

# Knuth's algorithm for generating Poisson samples
def _poisson(lmbd):
  L, k, p = math.exp(-lmbd), 0, 1
  while p > L:
    k += 1
    p *= random.uniform(0, 1)
  return max(k - 1, 0)

def _importance_sampling(pie, beta, action):
  sz = len(pie)
  # assert that the batch size of pie, beta, action is same
  assert sz == len(beta) and sz == len(action)
  rho = torch.ones((sz,1), dtype=torch.float32)
  for i in range(sz):
    rho[i] = (_multivariate_normal_pdf(action[i], pie[i]) / _multivariate_normal_pdf(action[i], beta[i])).detach()
  # rho = rho.clamp(min= 0.000001)
  return rho

# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr

def gaussianEntropy(mu, ln_var):
  return (0.5 * mu.data.shape[-1] * (np.log(2 * np.pi) + 1) + 0.5 *ln_var.sum(1)).mean(0)

# Updates networks
def _update_networks(args, T, model, shared_model, shared_average_model, loss, optimiser):
  # Zero shared and local grads
  optimiser.zero_grad()
  """
  Calculate gradients for gradient descent on loss functions
  Note that math comments follow the paper, which is formulated for gradient ascent
  """
  loss.backward()
  # Gradient L2 normalisation
  nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)

  # Transfer gradients to shared model and update
  _transfer_grads_to_shared_model(model, shared_model)
  optimiser.step()
  if args.lr_decay:
    # Linearly decay learning rate
    _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))

  # Update shared_average_model
  for shared_param, shared_average_param in zip(shared_model.parameters(), shared_average_model.parameters()):
    shared_average_param = args.trust_region_decay * shared_average_param + (1 - args.trust_region_decay) * shared_param

# Enables/disables gradients of model apart from policy head
def _isolate_policy_grads(model, isolate):
  for name, param in model.named_parameters():
    if 'fc_actor' not in name:
      param.requires_grad = not isolate

# Computes a trust region loss based on an existing loss and two distributions
def _trust_region_loss(model, distribution, ref_distribution, loss, threshold):
  # _isolate_policy_grads(model, True)  # Disable gradients for other parameters  
  sz = len(distribution)
  # Compute gradients from original loss
  model.zero_grad()
  loss.backward(retain_graph=True)

  # Gradients should be treated as constants (not using detach as volatility can creep in when double backprop is not implemented)
  g = [Variable(param.grad.data.clone()) for param in model.parameters() if param.grad is not None]
  model.zero_grad()

  _distribution, _ref_distribution = torch.ones((sz,1),dtype=torch.float32), torch.ones((sz,1),dtype=torch.float32)

  # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]

  assert distribution.shape == ref_distribution.shape
  for k in range(sz):
    policy = distribution[k]
    action_sample = (policy + math.sqrt(0.3)*torch.randn( policy.size() )).data
    # print (policy, action_sample)
    _distribution[k] = _multivariate_normal_pdf(action_sample, distribution[k]).clamp(min=0.000001)
    _ref_distribution[k] = _multivariate_normal_pdf(action_sample, ref_distribution[k]).clamp(min=0.000001)
  kl = (_ref_distribution * (_ref_distribution.log() - _distribution.log())).sum(1).mean(0)
  (-kl).backward(retain_graph=True)
  k = [Variable(param.grad.data.clone()) for param in model.parameters() if param.grad is not None]

  model.zero_grad()


  # Compute dot products of gradients
  k_dot_g = sum(torch.sum(k_p * g_p) for k_p, g_p in zip(k, g))
  k_dot_k = sum(torch.sum(k_p ** 2) for k_p in k)
  # Compute trust region update
  # To remove the warning, added item() to the expression.
  # if k_dot_k.data[0] > 0:
  # z = 0
  if k_dot_k.item() > 0:
    # z = 1
    trust_factor = ((k_dot_g - threshold) / k_dot_k).clamp(min=0)
  else:
    trust_factor = Variable(torch.zeros(1))
  # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
  z_star = [g_p - trust_factor.expand_as(k_p) * k_p for g_p, k_p in zip(g, k)]
  trust_loss = 0
  for param, z_star_p in zip(model.parameters(), z_star):
    trust_loss += (param * z_star_p).sum()
  # print ('done dana done')
  # _isolate_policy_grads(model, False)  # Re-enable gradients for other parameters
  return trust_loss


# Trains model
def _train(rank, args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions_dash, actions, rewards, Qret, average_policies, old_policies, loss_list):
  action_size = policies[0].size(-1)
  policy_loss, value_loss, entropy_loss = 0, 0, 0

  # Qret has the Value V(s(t+1)) OR 0.
  Qopc = Qret

  # Calculate n-step returns in forward view, stepping backwards from the last state
  t = len(rewards)
  for i in reversed(range(t)):
    curr_policy = policies[i]
    old_policy = old_policies[i]
    sz = len(curr_policy) # Size of batch

    # _multivariate_normal_pdf(actions[i], curr_policy) / _multivariate_normal_pdf(actions[i], old_policy)
    # Qopc ← r_i + γQopc
    Qopc = rewards[i] + args.discount * Qopc
    Qopc = Qopc.detach()
    
    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret
    Qret = Qret.detach()
    # Advantage due to retrace
    A = Qret - Vs[i]

    # Advantage due to opc
    Aopc = Qopc - Vs[i]

    # Advantage computed from model
    Amodel = Qs[i].detach() - Vs[i].detach()

    f_i_val = torch.ones((sz, 1), dtype=torch.float32)
    for k in range(sz):
      f_i_val[k] = _multivariate_normal_pdf(actions[i][k], curr_policy[k]).clamp(min=0.000001)

    # Log policy log(π(a_i|s_i; θ)) . The pie comes from distribution of multivariate Gaussian.  
    log_f = f_i_val.log()
    # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
    rho = _importance_sampling(curr_policy.detach(), old_policy.detach(), actions[i])
    rho_ = rho.clamp(max=args.trace_max).detach()
    single_step_policy_loss = (-rho_ * log_f * Aopc.detach()).mean(0)  # Average over batch

    rho_dash = _importance_sampling(curr_policy, old_policy, actions_dash[i])

    # Off-policy bias correction
    # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
    bias_weight = (1 - args.trace_max / rho_dash.detach()).clamp(min=0)

    f_idash_val = torch.ones((sz, 1), dtype=torch.float32)
    for k in range(sz):
      f_idash_val[k] = _multivariate_normal_pdf(actions_dash[i][k], curr_policy[k]).clamp(min=0.000001)
    log_f_idash = f_idash_val.log()

    # Averaging the loss for 1 step in the batch
    single_step_policy_loss -= (bias_weight * log_f_idash * Amodel).mean(0)

    if args.trust_region:
      # Policy update dθ ← dθ + ∂θ/∂θ∙z*
      policy_loss += _trust_region_loss(model, curr_policy, average_policies[i], single_step_policy_loss, args.trust_region_threshold)
    else:
      # Policy update dθ ← dθ + ∂θ/∂θ∙g
      policy_loss += single_step_policy_loss
  
    # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ)
    # entropy_loss -= args.entropy_weight * gaussianEntropy(curr_policy, acti)  # Sum over probabilities, average over batch

    value_loss += ((Qret - Qs[i]) ** 2 / 2).mean(0)  # Least squares loss

    # Calculating Ci for Q retrace
    truncated_rho = torch.pow(rho.clamp(max=1.).detach(), 1.0/action_size)

    # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qret = truncated_rho * (Qret.detach() - Qs[i].detach()) + Vs[i].detach()
    # Qret ← 1∙(Qopc - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qopc = (Qopc.detach() - Qs[i].detach()) + Vs[i].detach()
  if not loss_list[0] or T.value() - loss_list[0][-1] > 100:
    loss_list[0].append(T.value())
    loss_list[1].append(policy_loss + value_loss.item() + entropy_loss)
    loss_list[2].append(policy_loss)
    loss_list[3].append(value_loss.item())
    loss_list[4].append(entropy_loss)

  # Update networks
    _update_networks(args, T, model, shared_model, shared_average_model, policy_loss + value_loss + entropy_loss, optimiser)


def learn(memory, args, model, shared_model, shared_average_model, T, optimiser, rank, loss_list):
  for _ in range(_poisson(args.replay_ratio)):
    # Act and train off-policy for a batch of (truncated) episode
    # print (args.batch_size)
    trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)
    
    # Lists of outputs for training
    policies, Qs, Vs,actions_dash, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], [], []

    # Loop over trajectories (bar last timestep)
    for i in range(len(trajectories) - 1):

      # Unpack first half of transition
      state = Variable(torch.cat(tuple((trajectory.state for trajectory in trajectories[i])), dim=0))
      action = Variable(torch.Tensor([trajectory.action for trajectory in trajectories[i]])).unsqueeze(1)
      reward = Variable(torch.Tensor([trajectory.reward for trajectory in trajectories[i]])).unsqueeze(1)
      old_policy = Variable(torch.cat(tuple((trajectory.policy for trajectory in trajectories[i])), 0))

      policy, Q, V,action_dash = model(Variable(state))
      average_policy, _, _ ,_ = shared_average_model(Variable(state))

      # Save outputs for offline training
      [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions_dash, actions, rewards, average_policies, old_policies),
                                         (policy, Q, V, action_dash, action, reward, average_policy, old_policy))]

      # Unpack second half of transition
      next_state = torch.cat(tuple((trajectory.state for trajectory in trajectories[i + 1])), 0)
      done = Variable(torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1))

    # Do forward pass for all transitions
    _, _, Qret, _ = model(Variable(next_state))
    # Qret = 0 for terminal s, V(s_i; θ) otherwise
    Qret = ((1 - done) * Qret).detach()

    # Train the network off-policy
    _train(rank, args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs,
           actions_dash, actions, rewards, Qret, average_policies, old_policies, loss_list)

# Acts and trains model
def trainCont(rank, args, T, shared_model, shared_average_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env).unwrapped
  env.seed(args.seed + rank)
  model = ContinousActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()
  # Normalise memory capacity by number of training processes
  memory = EpisodicReplayMemory(args.memory_capacity // args.num_processes, args.max_episode_length)
  steps = []
  total_loss = []
  loss_actor = []
  loss_critic = []
  loss_entropy = []
  t = 1  # Thread step counter
  done = True  # Start new episode
  while T.value() <= args.T_max:
    while True:
      # Sync with shared model at least every t_max steps
      model.load_state_dict(shared_model.state_dict())
      # Get starting timestep
      t_start = t
      # Reset or pass on hidden state
      if done:
        # Reset environment and done flag
        state = state_to_tensor(env.reset())
        done, episode_length = False, 0

      while not done and t - t_start < args.t_max:
        # Calculate policy and values
        policy, Q, V, action= model(Variable(state))
        average_policy, _, _, _ = shared_average_model(Variable(state))

        next_state, reward, done, _ = env.step(action[0])
        next_state = state_to_tensor(next_state)
        # TODO Check clamp rewards
        # reward = args.reward_clip and min(max(reward, -2.0), 2.0) or reward  # Optionally clamp rewards
        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        episode_length += 1  # Increase episode counter

        # Save (beginning part of) transition for offline training
        memory.append(state, action, reward, policy.data)  # Save just tensors

        # Increment counters
        t += 1
        T.increment()

        # Update state
        state = next_state

      # Train the network when enough experience has been collected
      if len(memory) >= args.replay_start:
        # Sample a number of off-policy episodes based on the replay ratio
        learn(memory, args, model, shared_model, shared_average_model, T, optimiser, rank, [steps, total_loss, loss_actor, loss_critic, loss_entropy])
      if done:
        # Qret = 0 for terminal state
        Qret = Variable(torch.zeros(1, 1))   
        # Save terminal state for offline training
        memory.append(state, None, None, None)

      # Finish episode
      if done:
        break
    done = True
  plot_loss(steps, total_loss, args, 'total_loss', rank)
  plot_loss(steps, loss_actor, args, 'actor_loss', rank)
  plot_loss(steps, loss_critic, args, 'critic_loss', rank)
  plot_loss(steps, loss_entropy, args, 'entropy_loss', rank)
  del steps, total_loss, loss_critic, loss_actor, loss_entropy
  env.close()
