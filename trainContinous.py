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
from utils import state_to_tensor
import math
from time import sleep

# Knuth's algorithm for generating Poisson samples
def _multivariate_normal_pdf(x, mu, sigma=None):
  # Note that sigma is a 0.3 * diagnol matrix
  X = x - mu
  d = x.shape[-1]
  X = X**2
  X = X.sum()
  X = X*0.3
  f = torch.exp(-X)/( ((2*math.pi)**(1/d)) * 0.3)
  return f

def _poisson(lmbd):
  L, k, p = math.exp(-lmbd), 0, 1
  while p > L:
    k += 1
    p *= random.uniform(0, 1)
  return max(k - 1, 0)


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


# Computes a trust region loss based on an existing loss and two distributions
def _trust_region_loss(model, distribution, ref_distribution, loss, threshold):
  # Compute gradients from original loss
  model.zero_grad()
  loss.backward(retain_graph=True)
  # Gradients should be treated as constants (not using detach as volatility can creep in when double backprop is not implemented)
  g = [Variable(param.grad.data.clone()) for param in model.parameters() if param.grad is not None]
  model.zero_grad()

  # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
  # TODO : Check log()
  action_sample = MultivariateNormal(distribution.detach(), torch.eye(distribution.shape[-1])*0.3).sample()
  kl = F.kl_div(_multivariate_normal_pdf(action_sample, distribution).log(),_multivariate_normal_pdf(action_sample, ref_distribution), size_average=False)
  # Compute gradients from (negative) KL loss (increases KL divergence)
  (-kl).backward(retain_graph=True)
  k = [Variable(param.grad.data.clone()) for param in model.parameters() if param.grad is not None]
  model.zero_grad()

  # Compute dot products of gradients
  k_dot_g = sum(torch.sum(k_p * g_p) for k_p, g_p in zip(k, g))
  k_dot_k = sum(torch.sum(k_p ** 2) for k_p in k)
  # Compute trust region update
  if k_dot_k.data[0] > 0:
    trust_factor = ((k_dot_g - threshold) / k_dot_k).clamp(min=0)
  else:
    trust_factor = Variable(torch.zeros(1))
  # z* = g - max(0, (k^T∙g - δ) / ||k||^2_2)∙k
  z_star = [g_p - trust_factor.expand_as(k_p) * k_p for g_p, k_p in zip(g, k)]
  trust_loss = 0
  for param, z_star_p in zip(model.parameters(), z_star):
    trust_loss += (param * z_star_p).sum()
  return trust_loss


# Trains model
def _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, average_policies, old_policies=None):
  # TODO : Remove off Policy parameter
  off_policy = old_policies is not None
  action_size = policies[0].size(-1)
  policy_loss, value_loss = 0, 0
  # Qret has the Value V(s(t+1)).
  Qopc = Qret
  # Calculate n-step returns in forward view, stepping backwards from the last state
  t = len(rewards)
  for i in reversed(range(t)):
    # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
    curr_policy = policies[i]
    old_policy = old_policies[i]

    with torch.no_grad():
      rho = _multivariate_normal_pdf(actions[i], curr_policy.detach()) / _multivariate_normal_pdf(actions[i], old_policy.detach())
      
    # Qopc ← r_i + γQopc
    Qopc = rewards[i] + args.discount * Qopc
    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret

    # Advantage A ← Qret - V(s_i; θ)
    A = Qret - Vs[i]
    Aopc = Qopc - Vs[i]
    # Log policy log(π(a_i|s_i; θ))
    f_i_val = _multivariate_normal_pdf(actions[i], curr_policy.detach())
    log_f = f_i_val.log()
    # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
    # Use Aopc for the actor learning.
    single_step_policy_loss = -(rho).clamp(max=args.trace_max) * log_f * Aopc.detach().mean(0)  # Average over batch
    # Off-policy bias correction
    # No change here as the Q is estimated from the neural network
    # Initializing the vaue of Q
    # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
    with torch.no_grad():
      f_distribution = MultivariateNormal(curr_policy.detach(), torch.eye(policy_dash.shape[-1])*0.3)
      action_dash = f_distribution.sample()

    with torch.no_grad():
      rho_dash = _multivariate_normal_pdf(action_dash, curr_policy) / _multivariate_normal_pdf(action_dash, old_policy)
      bias_weight = (1 - args.trace_max / rho_dash).clamp(min=0.)

    f_idash_val = _multivariate_normal_pdf(action_dash, curr_policy.detach())
    single_step_policy_loss -= (bias_weight * f_idash_val.log() * (Qs[i].detach() - Vs[i].detach())).sum(1).mean(0)
    if args.trust_region: 
      # Policy update dθ ← dθ + ∂θ/∂θ∙z*
      policy_loss += _trust_region_loss(model, curr_policy, average_policies[i], single_step_policy_loss, args.trust_region_threshold)
    else:
      # Policy update dθ ← dθ + ∂θ/∂θ∙g
      policy_loss += single_step_policy_loss

    # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
    policy_loss -= args.entropy_weight * -(f_idash_val.log() * f_idash_val).sum(1).mean(0)  # Sum over probabilities, average over batch

    # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
    # This will be Qret. No changes here
    value_loss += ((Qret - Qs[i]) ** 2 / 2).mean(0)  # Least squares loss

    # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
    # Rho_dash comes into picture when off_policy
    with torch.no_grad():
      truncated_rho = (rho.clamp(max=1.))**(1/action_size)
    # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qret = truncated_rho * (Qret - Qs[i].detach()) + Vs[i].detach()
    # Qret ← 1∙(Qopc - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qopc = (Qopc - Q.detach()) + V.detach()
  # Update networks
  _update_networks(args, T, model, shared_model, shared_average_model, policy_loss + value_loss, optimiser)


# Acts and trains model
def trainCont(rank, args, T, shared_model, shared_average_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  model = ContinousActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()

  # Normalise memory capacity by number of training processes
  memory = EpisodicReplayMemory(args.memory_capacity // args.num_processes, args.max_episode_length)

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
        hx, avg_hx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
        cx, avg_cx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
        # Reset environment and done flag
        state = state_to_tensor(env.reset())
        done, episode_length = False, 0
      else:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Lists of outputs for training
      policies, Qs, Vs, actions, rewards, average_policies = [], [], [], [], [], []

      while not done and t - t_start < args.t_max:
        # Calculate policy and values
        # hx and cx are hidden states in the LSTM
        policy, Q, V, (hx, cx) = model(Variable(state), (hx, cx))
        average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(Variable(state), (avg_hx, avg_cx))

        # Sample action
        # action = policy.multinomial().data[0, 0]  # Graph broken as loss for stochastic action calculated manually
        action = MultivariateNormal(policy, torch.eye(policy.shape[-1])*0.3).sample()
        # print("m :", m,"\naction :",action)
        next_state, reward, done, _ = env.step(action)
        # print (action, next_state, reward, done)
        # sleep(4)
        next_state = state_to_tensor(next_state)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        episode_length += 1  # Increase episode counter

        # Save (beginning part of) transition for offline training
        # print(policy.data)
        # sleep(4)
        memory.append(state, action, reward, policy.data)  # Save just tensors
        # # Save outputs for online training
        # [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
        #                                    (policy, Q, V, Variable(torch.FloatTensor([[action]])), Variable(torch.Tensor([[reward]])), average_policy))]

        # Increment counters
        t += 1
        T.increment()

        # Update state
        state = next_state

      # Break graph for last values calculated (used for targets, not directly as model outputs)
      if done:
        # Qret = 0 for terminal s
        Qret = Variable(torch.zeros(1, 1))   
        # Save terminal state for offline training
        memory.append(state, None, None, None)
      else:
        # Qret = V(s_i; θ) for non-terminal s
        _, _, Qret, _ = model(Variable(state), (hx, cx))
        Qret = Qret.detach()

      # Train the network on-policy
      # _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, average_policies)

      # Finish episode
      if done:
        break

    # Train the network when enough experience has been collected
    if len(memory) >= args.replay_start:
      # Sample a number of off-policy episodes based on the replay ratio
      for _ in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for a batch of (truncated) episode
        trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)

        # Reset hidden state
        hx, avg_hx = Variable(torch.zeros(args.batch_size, args.hidden_size)), Variable(torch.zeros(args.batch_size, args.hidden_size))
        cx, avg_cx = Variable(torch.zeros(args.batch_size, args.hidden_size)), Variable(torch.zeros(args.batch_size, args.hidden_size))

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, old_policies, average_policies = [], [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
          # Unpack first half of transition
          print (trajectories[i])
          sleep(10)
          state = torch.cat((trajectory.state for trajectory in trajectories[i]), )
          action = Variable(torch.Tensor([trajectory.action for trajectory in trajectories[i]])).unsqueeze(1)
          reward = Variable(torch.Tensor([trajectory.reward for trajectory in trajectories[i]])).unsqueeze(1)
          old_policy = Variable(torch.cat((trajectory.policy for trajectory in trajectories[i]), 0))

          # Calculate policy and values
          policy, Q, V, (hx, cx) = model(Variable(state), (hx, cx))
          average_policy, _, _, (avg_hx, avg_cx) = shared_average_model(Variable(state), (avg_hx, avg_cx))

          # Save outputs for offline training
          [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies, old_policies),
                                             (policy, Q, V, action, reward, average_policy, old_policy))]

          # Unpack second half of transition
          next_state = torch.cat((trajectory.state for trajectory in trajectories[i + 1]), 0)
          done = Variable(torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1))

        # Do forward pass for all transitions
        _, _, Qret, _ = model(Variable(next_state), (hx, cx))
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()

        # Train the network off-policy
        _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs,
               actions, rewards, Qret, average_policies, old_policies=old_policies)
    done = True

  env.close()
