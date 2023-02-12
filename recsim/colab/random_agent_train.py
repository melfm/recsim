import numpy as np
import os
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch

from recsim.environments import interest_evolution
from recsim.environments import long_term_satisfaction
from recsim.agents import random_agent
from gym import spaces



env_config = {'slate_size': 1,
              'seed': 0,
              'num_candidates': 5,
              'resample_documents': True}

env = interest_evolution.create_environment(env_config)
#env = long_term_satisfaction.create_environment(env_config)
for key, space in env.observation_space['doc'].spaces.items():
  print(key, ':', space)

user_space = env.observation_space['user']
doc_space = env.observation_space['doc']
num_candidates = env_config['num_candidates']
doc_space_shape = spaces.flatdim(list(doc_space.spaces.values())[0])
# Use the longer of user_space and doc_space as the shape of each row.
obs_shape = (np.max([spaces.flatdim(user_space), doc_space_shape]),)[0]

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
env.reset()

hidden_dim = 64
obs_dim = obs_shape + doc_space_shape * num_candidates + 5
hidden_depth = 2
output_dim = env.action_space.nvec[0]

steps_done = 0
res_dir = 'results'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

num_episodes = 500
episode_durations = []
all_rewards = []
eps_reward = 0.0
action_space = spaces.MultiDiscrete(output_dim * np.ones((1,)))
agent = random_agent.RandomAgent(action_space, random_seed=0)

for i_episode in range(num_episodes):
    # Initialize the environment and state
    obs = env.reset()
    for t in count():
        # Select and perform an action
        action = agent.step(1, observation=obs)
        _, reward, done, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        eps_reward += reward.item()

        if done:
            if i_episode % 10 == 0:
                print('Eps ', i_episode, 'Reward', eps_reward)
            episode_durations.append(t + 1)
            all_rewards.append(eps_reward)
            eps_reward = 0.0
            break

print('Complete')
plt.plot(all_rewards)
plt.ylabel('Rewards')
file_name = res_dir + '/rand_policy_rewards_slate.png'
plt.savefig(file_name)
plt.clf()
plt.clf()
plt.plot(episode_durations)
plt.ylabel('Episode Durations')
file_name = res_dir + '/rand_policy_eps_duration.png'
plt.savefig(file_name)