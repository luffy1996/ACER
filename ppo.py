# -*- coding: utf-8 -*-
import math
import random
import gym
import torch
from torch import nn
from torch.nn import functional as F
import copy

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

# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr

# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
  for param, shared_param in zip(model.parameters(), shared_model.parameters()):
    if shared_param.grad is not None:
      return
    shared_param._grad = param.grad

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


# Trains model
def _train(args, T, model, shared_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, mu_policies, old_policies):
  action_size = policies[0].size(1)
  policy_loss, value_loss, entropy_loss = 0, 0, 0

  # Calculate n-step returns in forward view, stepping backwards from the last state
  t = len(rewards)
  counter = 0

  for i in reversed(range(t)):
    # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
    counter += 1
    rho = policies[i].detach() / old_policies[i]
    rho_prime = old_policies[i] / mu_policies[i]

    # Qret ← r_i + γQret
    Qret = rewards[i] + args.discount * Qret
    # Advantage A ← Qret - V(s_i; θ)
    Aret = Qret - Vs[i]
    Amodel = Qs[i] - Vs.expand_as(Qs[i])

    # Log policy log(π(a_i|s_i; θ))
    r = rho.gather(1, actions[i])
    policy_loss += rho_prime.gather(1, actions[i]) * (torch.min(r*A.detach(), torch.clamp(r, min=1-epsilon, max=1+epsilon)*A.detach()))
    # Calculating truncation loss
    bias_weight = (1 - args.trace_max / rho).clamp(min=0) * old_policies[i]
    truncated_loss = torch.min(rho*Amodel.detach(), torch.clamp(rho, min=1-epsilon, max=1-epsilon)*Amodel.detach())
    trucation_policy_loss += (bias_weight.detach() * truncated_loss).sum(1).mean(0)

    # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
    entropy_loss -= args.entropy_weight * -(policies[i].log() * policies[i]).sum(1).mean(0)  # Sum over probabilities, average over batch

    # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
    Q = Qs[i].gather(1, actions[i])
    value_loss += ((Qret - Q) ** 2 / 2).mean(0)  # Least squares loss

    # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
    truncated_rho = rho_prime.gather(1, actions[i]).clamp(max=1)
    # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
    Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()

    #update network
    if counter==50:
      _update_networks(args, T, model, shared_model, (policy_loss + value_loss + entropy_loss)/counter, optimiser)
      counter, policy_loss, value_loss, entropy_loss = 0, 0, 0, 0

  # Update networks
  if counter!=0:
    _update_networks(args, T, model, shared_model, (policy_loss + value_loss + entropy_loss)/counter, optimiser)


def trainSEPPO(rank, args, T, shared_model, optimiser):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env)
  env.seed(args.seed + rank)
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()

  # Normalise memory capacity by number of training processes
  memory = EpisodicReplayMemory(args.memory_capacity // args.num_processes, args.max_episode_length)

  t = 1  # Thread step counter
  done = True  # Start new episode

  while T.value() <= args.T_max:
    # On-policy episode loop
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
        policy, Q, V = model(state)

        # Sample action
        action = torch.multinomial(policy, 1)[0, 0]

        # Step
        next_state, reward, done, _ = env.step(action.item())
        next_state = state_to_tensor(next_state)
        reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
        done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
        episode_length += 1  # Increase episode counter

        # Save (beginning part of) transition for offline training
        memory.append(state, action, reward, policy.detach())  # Save just tensors
        # Save outputs for online training
        # TODO : Check if online
        # [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, average_policies),
        #                                    (policy, Q, V, torch.LongTensor([[action]]), torch.Tensor([[reward]]), average_policy))]

        # Increment counters
        t += 1
        T.increment()

        # Update state
        state = next_state

      # Break graph for last values calculated (used for targets, not directly as model outputs)
      if done:
        # Qret = 0 for terminal s
        # Qret = torch.zeros(1, 1)
        # Save terminal state for offline training
        memory.append(state, None, None, None)
      else:
        # Qret = V(s_i; θ) for non-terminal s
        # _, _, Qret= model(state)
        # Qret = Qret.detach()

      # Train the network on-policy
      # _train(args, T, model, shared_model, shared_average_model, optimiser, policies, Qs, Vs, actions, rewards, Qret, average_policies)

      # Finish on-policy episode
      if done:
        break

    # Train the network off-policy when enough experience has been collected
    if len(memory) >= args.replay_start:
      # Sample a number of off-policy episodes based on the replay ratio
      old_model = copy.deepcopy(model)
      for _ in range(_poisson(args.replay_ratio)):
        # Act and train off-policy for a batch of (truncated) episode
        trajectories = memory.sample_batch(args.batch_size, maxlen=args.t_max)

        # Lists of outputs for training
        policies, Qs, Vs, actions, rewards, mu_policies= [], [], [], [], [], []

        # Loop over trajectories (bar last timestep)
        for i in range(len(trajectories) - 1):
          # Unpack first half of transition
          state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i]), 0)
          action = torch.LongTensor([trajectory.action for trajectory in trajectories[i]]).unsqueeze(1)
          reward = torch.Tensor([trajectory.reward for trajectory in trajectories[i]]).unsqueeze(1)
          mu_policy = torch.cat(tuple(trajectory.policy for trajectory in trajectories[i]), 0)

          # Compute the old policy
          old_policy, Q_old, V_old = old_model(state)

          # Calculate policy and values
          policy, _, _ = model(state)

          # Save outputs for offline training
          [arr.append(el) for arr, el in zip((policies, Qs, Vs, actions, rewards, mu_policies, old_policies),
                                             (policy, Q_old, V_old, action, reward, mu_policy, old_policy))]

          # Unpack second half of transition
          next_state = torch.cat(tuple(trajectory.state for trajectory in trajectories[i + 1]), 0)
          done = torch.Tensor([trajectory.action is None for trajectory in trajectories[i + 1]]).unsqueeze(1)

        # Do forward pass for all transitions
        _, _, Qret = old_model(next_state)
        # Qret = 0 for terminal s, V(s_i; θ) otherwise
        Qret = ((1 - done) * Qret).detach()

        # Train the network off-policy
        _train(args, T, model, shared_model, optimiser, policies, Qs, Vs,
               actions, rewards, mu_policies=mu_policies, old_policies=old_policies)
    done = True

  env.close()
