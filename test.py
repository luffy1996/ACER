# -*- coding: utf-8 -*-
import time
from datetime import datetime
import gym
import torch
from torch.autograd import Variable
from model import ContinousActorCritic
from utils import state_to_tensor, plot_line
import csv
import math
from time import sleep
def test(rank, args, T, shared_model):
  torch.manual_seed(args.seed + rank)

  env = gym.make(args.env).unwrapped
  env.seed(args.seed + rank)
  model = ContinousActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.eval()

  can_test = True  # Test flag
  t_start = 1  # Test step counter to check against global counter
  rewards, steps = [], []  # Rewards and steps for plotting
  l = str(len(str(args.T_max)))  # Max num. of digits for logging steps
  done = True  # Start new episode

  while T.value() <= args.T_max:
    if can_test:
      t_start = T.value()  # Reset counter

      # Evaluate over several episodes and average results
      avg_rewards, avg_episode_lengths = [], []
      for _ in range(args.evaluation_episodes):
        while True:
          # Reset or pass on hidden state
          if done:
            # Sync with shared model every episode
            model.load_state_dict(shared_model.state_dict())
            hx = torch.zeros(1, args.hidden_size)
            cx = torch.zeros(1, args.hidden_size)
            # Reset environment and done flag
            state = state_to_tensor(env.reset())
            done, episode_length = False, 0
            reward_sum = 0

          # Optionally render validation states
          if args.render:
            env.render()

          # Calculate policy
          with torch.no_grad():
            policy, _, _, _, (hx, cx) = model(Variable(state), (hx, cx))  # Break graph for memory efficiency

          # Choose action greedily
          action = policy
          # Step
          # if math.isnan(action):
          #   print (action)
          #   print (model(Variable(state)))
          #   print (state)
          #   sleep(10)

          action = action.clamp(min=-2.0,max=2.0) 
          state, reward, done, _ = env.step(action[0])
          state = state_to_tensor(state)
          # if (math.isnan(reward)):
          #   print (state, done, action)
          #   sleep(10)
          # print (reward)
          reward_sum += reward
          done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
          episode_length += 1  # Increase episode counter

          # Log and reset statistics at the end of every episode
          if done:
            avg_rewards.append(reward_sum)
            avg_episode_lengths.append(episode_length)
            break
      average_rewards = sum(avg_rewards) / args.evaluation_episodes
      average_episode_lengths = sum(avg_episode_lengths) / args.evaluation_episodes
      # print (sum(avg_episode_lengths) / args.evaluation_episodes)
      print('Time : ', datetime.now(),  
            'Step : ', t_start,
            'Avg. Reward : ', sum(avg_rewards) / args.evaluation_episodes,
            'Avg. Steps Per Episode :' , sum(avg_episode_lengths) / args.evaluation_episodes)
      # print ('average steps : ', average_episode_lengths)
      fields = [t_start, sum(avg_rewards) / args.evaluation_episodes, sum(avg_episode_lengths) / args.evaluation_episodes, str(datetime.now())]
      with open('results/'+args.name+'/test_results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields) 
      if args.evaluate:
        return

      rewards.append(avg_rewards)  # Keep all evaluations
      steps.append(t_start)
      plot_line(steps, rewards, args)  # Plot rewards
      torch.save(model.state_dict(), 'results/'+args.name+'/model.pth')  # Save model params
      can_test = False  # Finish testing
    else:
      if T.value() - t_start >= args.evaluation_interval:
        can_test = True

    time.sleep(0.001)  # Check if available to test every millisecond

  env.close()
