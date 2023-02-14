import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import count

import torch
from gym import spaces

from recsim.environments import interest_evolution
from recsim.environments import long_term_satisfaction


batch_size=50
lr=0.001
episilon=0.5
replay_memory_size=10000
gamma=0.9
target_update_iter=100
log_internval=10
train_episodes = 1000

env_type = 'evolve'
env_config = {'slate_size': 3,
              'seed': 0,
              'num_candidates': 10,
              'resample_documents': True}

if env_type == 'evolve':
    env = interest_evolution.create_environment(env_config)
elif env_type == 'longterm':
    # TODO : how does this env work with the same config?
    # is it not the choc vs kale?
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
# n_action=env_config['slate_size']
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
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2=torch.nn.Linear(hidden,hidden)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out=torch.nn.Linear(hidden,n_action)
        self.out.weight.data.normal_(0, 0.1)


    def forward(self,x):
        x=self.fc1(x)
        x=torch.nn.functional.relu(x)
        x=self.fc2(x)
        x=torch.nn.functional.relu(x)
        out=self.out(x)
        return out

class replay_memory():
    def __init__(self):
        self.memory_size=replay_memory_size
        self.memory=np.array([])
        self.cur=0
        self.new=0
    def size(self):
        return self.memory.shape[0]

    def store_transition(self,trans):
        # trans : [s,a,r,s',done]
        if(self.memory.shape[0]<self.memory_size):
            if self.new==0:
                self.memory=np.array(trans)
                self.new=1
            elif self.memory.shape[0]>0:
                self.memory=np.vstack((self.memory,trans))

        else:
            self.memory[self.cur,:]=trans
            self.cur=(self.cur+1)%self.memory_size

    def sample(self):
        if self.memory.shape[0]<batch_size:
            return -1
        sam=np.random.choice(self.memory.shape[0],batch_size)
        return self.memory[sam]

class DQN():

    def __init__(self):
        self.eval_q_net,self.target_q_net=net().to(device),net().to(device)
        self.replay_mem=replay_memory()
        self.iter_num=0
        self.optimizer=torch.optim.Adam(self.eval_q_net.parameters(),lr=lr)
        self.loss=torch.nn.MSELoss().to(device)

    def choose_action(self,qs):
        if np.random.uniform()<episilon:
            return torch.topk(qs,k=n_action).indices
        else:
            return torch.randint(0, n_action ,(n_action,))

    def greedy_action(self,qs):
        return torch.topk(qs,k=n_action).indices

    def learn(self):
        if(self.iter_num%target_update_iter==0):
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num+=1

        batch=self.replay_mem.sample()
        b_s=torch.FloatTensor(batch[:,0].tolist()).to(device)
        b_a=torch.LongTensor(np.array(batch[:,1].tolist())).to(device)
        b_r=torch.FloatTensor(batch[:,2].tolist()).to(device)
        b_s_=torch.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=torch.FloatTensor(batch[:,4].tolist()).to(device)
        q_target=torch.zeros((batch_size,n_action)).to(device)
        q_eval=self.eval_q_net(b_s)
        q_eval=torch.gather(q_eval,dim=1,index=b_a)
        q_next=self.target_q_net(b_s_).detach()
        for i in range(b_d.shape[0]):
            if(int(b_d[i].tolist()[0])==0):
                q_target[i]=b_r[i]+gamma*torch.unsqueeze(torch.max(q_next[i],0)[0],0)
            else:
                q_target[i]=b_r[i]
        td_error=self.loss(q_eval,q_target)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()
        return td_error.item()

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

dqn=DQN()
all_rewards = []
all_losses = []
all_actions = []

for episode in range(train_episodes):
    state = env.reset()
    state = flatten_obs(state)
    reward = 0.0
    done = False
    for t in count():
        state_tensor = torch.FloatTensor(state).to(device)
        state_normalized = torch.nn.functional.normalize(state_tensor, dim=0)
        qvals = dqn.eval_q_net(state_normalized)
        action = dqn.choose_action(qvals)
        action = action.cpu().numpy()

        next_state,reward,done,_ = env.step(action)

        next_state = flatten_obs(next_state)
        assert(not(np.array_equal(state, next_state)))
        transition=[state.tolist(),action,[reward],next_state.tolist(),[done]]
        dqn.replay_mem.store_transition(transition)

        state = next_state
        if dqn.replay_mem.size()>batch_size:
            loss = dqn.learn()
            all_losses.append(loss)
        if done:
            done = False
            break
    if episode%log_internval==0:
        # Evaluation
        total_reward=0.0
        for i in range(10):
            eval_state = env.reset()
            eval_state_original = eval_state
            eval_state = flatten_obs(eval_state)
            eval_reward = 0.0
            eps_reward = 0.0
            eval_step = 0
            edone = False
            for t in count():
                eval_qvals = dqn.eval_q_net(torch.FloatTensor(eval_state).to(device))
                eval_action = dqn.greedy_action(eval_qvals).cpu().numpy()
            
                eval_next_state, eval_reward, edone,_ = env.step(eval_action)

                eval_next_state_original = eval_next_state
                eval_next_state = flatten_obs(eval_next_state)
                eps_reward+=eval_reward
                if edone:
                    edone = False
                    # print('Episode duration ', t)
                    break
                eval_state=eval_next_state
                eval_state_original = eval_next_state_original
            total_reward+=eps_reward

        print("Episode:"+format(episode)+", eval reward:"+format(total_reward/10))
        all_rewards.append(total_reward/10)
        #print('Selected actioom', action)
        all_actions.append(eval_action)

print('Complete')
plt.plot(all_rewards)
plt.ylabel('Rewards')
file_name = res_dir + '/DQN_recsim_larger_{}_rewards.png'.format(env_type)
plt.savefig(file_name)

plt.clf()

plt.plot(all_losses)
plt.ylabel('TD_Error')
file_name = res_dir + '/DQN_recsim_larger_{}_losses.png'.format(env_type)
plt.savefig(file_name)

plt.clf()

plt.plot(all_actions)
plt.ylabel('Action_Selection')
file_name = res_dir + '/DQN_recsim_larger_{}_actions.png'.format(env_type)
plt.savefig(file_name)

print('Reward average ', np.mean(all_rewards))