#!/usr/bin/env python
import functools
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

from recsim import choice_model
from recsim.simulator import environment
from recsim.environments import interest_evolution
from recsim.environments import interest_exploration
from recsim.agents import random_agent


#############################
# Exploration Environment
#############################
env_config = {'slate_size': 2,
              'seed': 0,
              'num_candidates': 15,
              'resample_documents': True}

# What is the difference between exploration and evolution envs?
# Which one is more appropriate to train with DQN ?
ie_environment = interest_exploration.create_environment(env_config)
initial_observation = ie_environment.reset()

print('User Observable Features')
print(initial_observation['user'])
print('User Response')
print(initial_observation['response'])
print('Document Observable Features')
for doc_id, doc_features in initial_observation['doc'].items():
  print('ID:', doc_id, 'features:', doc_features)


print('Document observation space')
for key, space in ie_environment.observation_space['doc'].spaces.items():
  print(key, ':', space)
print('Response observation space')
print(ie_environment.observation_space['response'])
print('User observation space')
print(ie_environment.observation_space['user'])


slate = [0, 1]
for slate_doc in slate:
  print(list(initial_observation['doc'].items())[slate_doc])

# Step with a random slate ?
# Who steps? env or agent??
observation, reward, done, _ = ie_environment.step(slate)

env_act_space = ie_environment.action_space
print('Environment action space ', env_act_space)
# Create a random agent
num_candidates = env_config['num_candidates']
slate_size = env_config['slate_size']
action_space = spaces.MultiDiscrete(num_candidates * np.ones((slate_size,)))
agent = random_agent.RandomAgent(action_space, random_seed=0)

slate = agent.step(reward, observation)
print('Explor: Recommended slate ', slate)
slate = agent.step(reward, observation)
print('Explor: Recommended slate ', slate)
slate = agent.step(reward, observation)
print('Explor: Recommended slate ', slate)

#############################
# Exploration Environment
#############################
# Create a candidate_set with 5 items
# Create a simple user
slate_size = 2
user_model = interest_evolution.IEvUserModel(
    slate_size,
    choice_model_ctor=choice_model.MultinomialLogitChoiceModel,
    response_model_ctor=interest_evolution.IEvResponse)

num_candidates = 5
document_sampler = interest_evolution.IEvVideoSampler()
ievsim = environment.Environment(user_model, document_sampler,
                                    num_candidates, slate_size)

# Create agent
action_space = spaces.MultiDiscrete(num_candidates * np.ones((slate_size,)))
agent = random_agent.RandomAgent(action_space, random_seed=0)

# This agent doesn't use the previous user response
observation, documents = ievsim.reset()
slate = agent.step(1, dict(user=observation, doc=documents))
print('Evolution: Recommended slate ', slate)