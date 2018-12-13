# -*- coding: utf-8 -*-
import math
import random
import gym
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.sampler import BatchSampler, RandomSampler
from time import sleep

from memory import EpisodicReplayMemory
from model import ActorCritic
from utils import state_to_tensor


# Knuth's algorithm for generating Poisson samples
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
def _update_networks(args, T, model, shared_model, loss, optimiser):
  # Zero shared and local grads
  optimiser.zero_grad()
  """
  Calculate gradients for gradient descent on loss functions
  Note that math comments follow the paper, which is formulated for gradient ascent
  """
  loss.backward(retain_graph=True)
  # Gradient L2 normalisation
  nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)

  # Transfer gradients to shared model and update
  _transfer_grads_to_shared_model(model, shared_model)
  optimiser.step()
  if args.lr_decay:
    # Linearly decay learning rate
    _adjust_learning_rate(optimiser, args.lr * (1 - T.value() / args.T_max))

# Trains model
def _train(args, T, model, shared_model, optimiser, actions, rewards, dones, qret, behaviour_policies, old_policies, states, on_policy):

  action_size = behaviour_policies[0].size(1)
  # Calculate n-step returns in forward view, stepping backwards from the last state
  total_epoches = 1 if on_policy else args.epoches
  # print (total_epoches)
  for _ in range(total_epoches):
    t = len(rewards)
    policy_losses, entropy_losses, value_losses = [], [], []
    # Evaluate the policies
    hx, cx = None, None
    policies, Qs, Vs = [], [], []
    if on_policy:
      hx = torch.zeros(args.batch_size, args.hidden_size)
      cx = torch.zeros(args.batch_size, args.hidden_size)
      
      for i in range(t):
        policy ,Q, V, (hx, cx) = model(states[i], (hx, cx))
        [arr.append(el) for arr, el in zip((Qs, Vs),
                                           (Q, V))]
        policies.append(policy)
        hx = hx*(1-dones[i])
        cx = cx*(1-dones[i])
    else:
      hx = torch.zeros(1, args.hidden_size)
      cx = torch.zeros(1, args.hidden_size)
      policies = []
      for i in range(t):
        policy ,Q ,V , (hx, cx) = model(states[i], (hx, cx))
        policies.append(policy)
        [arr.append(el) for arr, el in zip((Qs, Vs),
                                           (Q, V))]
        hx = hx*(1-dones[i])
        cx = cx*(1-dones[i])
    Qret = qret
    for i in reversed(range(t)):
      # Importance sampling weights ρ ← beta(∙|s_i) / pie_old(∙|s_i); 1 for ff-policy
      rho = (old_policies[i])/ (behaviour_policies[i] + args.epsilon)

      # Qret ← r_i + γQret
      Qret = rewards[i] + args.discount * Qret * (1 - dones[i])
      # Advantage A ← Qret - V(s_i; θ)
      A = Qret - Vs[i]

      r_theta = (policies[i])/(old_policies[i] + args.epsilon)

      # Calculate surr1 and surr2
      eps = args.seppo_clip_param * (1 - T.value() / args.T_max)
      surr1 = r_theta * A
      surr2 = torch.clamp(r_theta, 1. - eps,
                          1. + eps) * A

      # check : If bias trucation needed.
      if on_policy:
        single_step_policy_loss_0 = ((torch.clamp(rho.gather(1, actions[i]), max=args.max_seppo_rho_value))*(-torch.min(surr1.gather(1, actions[i]), surr2.gather(1, actions[i])))).mean()
        single_step_policy_loss_1 = (torch.clamp(1. - args.max_seppo_rho_value/rho, min=0., max=1.)*(-torch.min(surr1, surr2))).mean()
        single_step_policy_loss = single_step_policy_loss_0 + single_step_policy_loss_1
      else:
        single_step_policy_loss = (-torch.min(surr1, surr2)).mean()
      single_step_entropy_loss = -args.entropy_weight*-(policies[i].log() * policies[i]).sum(1).mean(0)  # Sum over probabilities, average over batch

      # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
      # print (Qs[i])
      Q = Qs[i].gather(1, actions[i])
      # print (Q, type(Q), Qret, type(Qret))
      # sleep(100000)
      assert Q.size() == Qret.size()
      singe_step_value_loss = args.value_weight*((Qret - Q) ** 2 / 2).mean(0)  # Least squares loss
      [arr.append(el) for arr, el in zip((policy_losses, entropy_losses, value_losses),
                                         (single_step_policy_loss, single_step_entropy_loss, singe_step_value_loss))]

      # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
      rho_new = ((policies[i] + args.epsilon)/ (behaviour_policies[i] + args.epsilon)).detach()

      truncated_rho = rho_new.gather(1, actions[i]).clamp(max=1)
      # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
      Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()
      
    assert len(policy_losses) == len(entropy_losses)
    assert len(policy_losses) == len(value_losses)

    if on_policy:
      random_sampler = RandomSampler(policy_losses) # We only need the length. Hence just send the policy_losses.
      batch_sampler = BatchSampler(random_sampler, args.train_batch_size, True) # drop last is false
      # Generators for batch sanpling
      # print (policy_losses)
      for i in batch_sampler:
        # print (i)
        policy_loss, entropy_loss, value_loss = [policy_losses[j] for j in i], [entropy_losses[j] for j in i], [value_losses[j] for j in i]
        batch_loss = sum(policy_loss + entropy_loss + value_loss)/len(policy_loss)
        _update_networks(args, T, model, shared_model, batch_loss, optimiser)
    else:
        batch_loss = sum(policy_losses + entropy_losses + value_losses)/len(policy_losses)
        _update_networks(args, T, model, shared_model, batch_loss, optimiser)
# Acts and trains model
def train(rank, args, T, shared_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()

  if not args.on_policy:
    # Normalise memory capacity by number of training processes
    memory = EpisodicReplayMemory(args.memory_capacity // args.num_processes, args.t_max)

  t = 1  # Thread step counter
  t_episode_start = t
  done = True  # Start new episode

  while T.value() <= args.T_max:
    # Sync with shared model at least every t_max steps
    model.load_state_dict(shared_model.state_dict())
    # Get starting timestep
    t_start = t

    # Reset or pass on hidden state
    if done:
      # check
      hx = torch.zeros(1, args.hidden_size)
      cx = torch.zeros(1, args.hidden_size)
      # Reset environment and done flag
      state = state_to_tensor(env.reset())
      done, episode_length = False, 0
    else:
      # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
      hx = hx.detach()
      cx = cx.detach()

    ## Learn On Policy
    actions, rewards, dones, behaviour_policies, old_policies, states = [], [], [], [], [], []

    while t - t_start < args.t_max:
      # Calculate policy and values
      policy, _, _, (hx, cx) = model(state, (hx, cx))

      # Sample action
      action = torch.multinomial(policy, 1)[0, 0]

      # Step
      next_state, reward, done, _ = env.step(action.item())
      next_state = state_to_tensor(next_state)
      reward = -10 if done else reward  # Optionally clamp rewards
      episode_length += 1  # Increase episode counter
      done = done or t - t_episode_start == args.max_episode_length
      # Save (beginning part of) transition for offline training
      old_policy = policy.detach()
      behaviour_policy = policy.detach()
      memory.append(state, action, reward, policy.detach(), done)  # Save just tensors
      [arr.append(el) for arr, el in zip((actions, rewards, dones,behaviour_policies, old_policies, states),
                                         (torch.LongTensor([[action]]), torch.Tensor([[reward]]), done, behaviour_policy, old_policy, state))]
      # Update state
      state = next_state

      # Special case for done
      if done:
        # check
        if t - t_episode_start == args.max_episode_length:
          t_episode_start = t
        hx = torch.zeros(1, args.hidden_size)
        cx = torch.zeros(1, args.hidden_size)
        # Reset environment and done flag
        state = state_to_tensor(env.reset())
        done, episode_length = False, 0
      else:
        # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
        hx = hx.detach()
        cx = cx.detach()

      # Increment counters
      t += 1

      T.increment()
    # Do forward pass for all transitions
    _, _, Qret, _ = model(next_state, (hx, cx))
    # Qret = 0 for terminal s, V(s_i; θ) otherwise
    Qret = ((1 - done) * Qret).detach()
    _train(args, T, model, shared_model, optimiser,
           actions, rewards, dones, Qret, behaviour_policies, old_policies, states, False)
    
    # Train the network off-policy when enough experience has been collected
    if len(memory) >= args.replay_start:
      # Sample a number of off-policy episodes based on the replay ratio
      for _ in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for a batch of (truncated) episode
        trajectories = memory.sample_batch(args.batch_size)

        hx = torch.zeros(args.batch_size, args.hidden_size)
        cx = torch.zeros(args.batch_size, args.hidden_size)

        # Lists of outputs for training
        old_policies, actions, rewards, dones, behaviour_policies, states = [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
          # Unpack first half of transition
          state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i]), 0)
          action = torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1)
          reward = torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1)
          behaviour_policy = torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0)
          done = torch.Tensor([trajectory.done for trajectory in trajectories[i]]).unsqueeze(1)

          # Calculate policy and values
          policy, _, _, (hx, cx) = model(state, (hx, cx))
          # Old policy is the detached policy
          old_policy = policy.detach()
          # Save outputs for offline training
          [arr.append(el) for arr, el in zip((states, actions, rewards, dones, old_policies, behaviour_policies),
                                             (state, action, reward, done, old_policy, behaviour_policy))]

          # Unpack second half of transition
          next_state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i + 1]), 0)
          hx = (1-done)*hx
          cx = (1-done)*cx
        _, _, Qret, _ = model(next_state, (hx, cx))
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()
        # Train for Epoches
        _train(args, T, model, shared_model, optimiser,
               actions, rewards, dones, Qret, behaviour_policies, old_policies, states, True)
    done = True

  env.close()
