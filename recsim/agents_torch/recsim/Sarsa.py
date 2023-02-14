import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import count

import torch
from gym import spaces

from recsim.environments import interest_evolution
from recsim.environments import long_term_satisfaction

import pdb

gamma=0.9
episilon=0.9
lr=0.001
target_update_iter=100
log_internval=100
train_episodes = 10000
terminal_step = 300

env_type = 'evolve'
env_config = {'slate_size': 1,
              'seed': 0,
              'num_candidates': 10,
              'resample_documents': True}

if env_type == 'evolve':
    env = interest_evolution.create_environment(env_config)
elif env_type == 'longterm':
    env = long_term_satisfaction.create_environment(env_config)
else:
    raise ValueError('Invalid env.')

for key, space in env.observation_space['doc'].spaces.items():
  print(key, ':', space)

user_space = env.observation_space['user']
doc_space = env.observation_space['doc']
num_candidates = env_config['num_candidates']
doc_space_shape = spaces.flatdim(list(doc_space.spaces.values())[0])
# Use the longer of user_space and doc_space as the shape of each row.
usr_obs_shape = (np.max([spaces.flatdim(user_space), doc_space_shape]),)[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_action = env.action_space.nvec.shape[0]
n_state=usr_obs_shape + doc_space_shape * num_candidates + 5
hidden=32
print('Action space ', n_action)
print('State obs dim ', n_state)

res_dir = 'results'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

class net(torch.nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1=torch.nn.Linear(n_state,hidden)
        self.out=torch.nn.Linear(hidden,n_action)
    

    def forward(self,x):
        x=self.fc1(x)
        x=torch.nn.functional.relu(x)
        out=self.out(x)
        return out

class Sarsa():
    def __init__(self):
        self.net,self.target_net=net(),net()
        self.iter_num=0
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=lr)
        self.loss=torch.nn.MSELoss().to(device)

    def learn(self,s,a,s_,r,done):
        #pdb.set_trace()
        eval_q=self.net(torch.Tensor(s))[a]
        target_q=self.target_net(torch.FloatTensor(s_))
        target_a=self.choose_action(target_q)
        target_q=target_q[target_a]
        if not done:
            y=gamma*target_q+r
        else:
            # Not sure if this is what we want
            y=torch.ones(n_action) * r
        #loss=(y-eval_q)**2
        loss = self.loss(y, eval_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iter_num+=1
        #pdb.set_trace()
        if self.iter_num%10==0:
            self.target_net.load_state_dict(self.net.state_dict())
        return target_a, loss.item()

    def greedy_action(self,qs):
        return torch.topk(qs,k=n_action).indices

    def random_action(self):
        return torch.randint(0, n_action ,(n_action,))

    def choose_action(self,qs):
        if np.random.rand()>episilon:
            return self.random_action()
        else:
            return self.greedy_action(qs)

def flatten_obs(obs):

    obs_usr = obs['user']
    obs_docs = obs['doc']
    response = obs['response']
    all_docs = []
    all_responses = []
    for _, item in obs_docs.items():
        all_docs.append(item)
    if response is not None:
        for _, item in response[0].items():
            all_responses.append(item)
    else:
        response_pad = np.zeros((5), dtype=np.float64)
        all_responses.append(response_pad)
    # TODO : This is ugly fix it.
    all_responses = np.array(all_responses).flatten()

    all_docs_np = np.array(all_docs).flatten()
    obs_flatten = np.concatenate((obs_usr, all_docs_np, all_responses), axis=0)
    return obs_flatten

sarsa=Sarsa()
all_rewards = []
all_losses = []

for episode in range(train_episodes):
    state = env.reset()
    state = flatten_obs(state)
    reward = 0.0
    done = False
    qvals = sarsa.net(torch.Tensor(state))
    action = sarsa.choose_action(qvals)
    action = action.cpu().numpy()
    #import pdb;pdb.set_trace()
    for t in count():
        next_state,reward,done,_ = env.step(action)
        next_state = flatten_obs(next_state)
        action, loss = sarsa.learn(state, action, next_state, reward, done)
        #print('Loss ', loss)
        state = next_state
        all_losses.append(loss)
        if done:
            break
    if episode%log_internval==0:
        # Evaluation
        total_reward=0.0
        for i in range(10):
            eval_state = env.reset()
            eval_state = flatten_obs(eval_state)
            eval_reward = 0.0
            eps_reward = 0.0
            edone = False
            for t in count():
                eval_qvals = sarsa.net(torch.Tensor(eval_state))
                action = sarsa.greedy_action(eval_qvals)
                eval_next_state, eval_reward, edone,_=env.step(action.tolist())
                eval_next_state = flatten_obs(eval_next_state)
                eps_reward+=eval_reward
                if edone:
                    edone = False
                    break
                eval_state=eval_next_state
            total_reward+=eps_reward
        total_avg_reward = total_reward/10
        print("Episode:"+format(episode)+", eval reward:"+format(total_avg_reward))
        all_rewards.append(total_avg_reward)


print('Complete')
plt.plot(all_rewards)
plt.ylabel('Rewards')
file_name = res_dir + '/Sarsa_recsim_larger_{}_rewards.png'.format(env_type)
plt.savefig(file_name)

plt.clf()

plt.plot(all_losses)
plt.ylabel('TD_Error')
file_name = res_dir + '/Sarsa_recsim_larger_{}_loss.png'.format(env_type)
plt.savefig(file_name)


print('Reward average ', np.mean(all_rewards))
