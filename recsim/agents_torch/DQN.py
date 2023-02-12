import numpy as np
import os
from itertools import count
import matplotlib.pyplot as plt

import gym
import torch

batch_size=50
lr=0.001
episilon=0.9
replay_memory_size=10000
gamma=0.9
target_update_iter=100
log_internval=100

env=gym.make('CartPole-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]
hidden=32

res_dir = 'results'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

class net(torch.nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1=torch.nn.Linear(n_state,hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out=torch.nn.Linear(hidden,n_action)
        self.out.weight.data.normal_(0, 0.1)


    def forward(self,x):
        x=self.fc1(x)
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
            return torch.argmax(qs).tolist()
        else:
            return np.random.randint(0,n_action)

    def greedy_action(self,qs):
        return torch.argmax(qs)

    def learn(self):
        if(self.iter_num%target_update_iter==0):
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num+=1

        batch=self.replay_mem.sample()
        b_s=torch.FloatTensor(batch[:,0].tolist()).to(device)
        b_a=torch.LongTensor(batch[:,1].astype(int).tolist()).to(device)
        b_r=torch.FloatTensor(batch[:,2].tolist()).to(device)
        b_s_=torch.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=torch.FloatTensor(batch[:,4].tolist()).to(device)
        q_target=torch.zeros((batch_size,1)).to(device)
        q_eval=self.eval_q_net(b_s)
        q_eval=torch.gather(q_eval,dim=1,index=torch.unsqueeze(b_a,1))
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

dqn=DQN()
all_rewards = []

for episode in range(10000):
    state = env.reset()
    reward = 0.0
    for t in count():
        qvals = dqn.eval_q_net(torch.FloatTensor(state).to(device))
        action = dqn.choose_action(qvals)

        next_state,reward,done,_ = env.step(action)
        transition=[state.tolist(),action,[reward],next_state.tolist(),[done]]
        dqn.replay_mem.store_transition(transition)

        state = next_state
        if dqn.replay_mem.size()>batch_size:
            dqn.learn()
        if done:
            break
    if episode%log_internval==0:
        # Evaluation
        total_reward=0.0
        for i in range(10):
            eval_state = env.reset()
            eval_reward = 0.0
            eps_reward = 0.0
            time=0
            for t in count():
                eval_qvals = dqn.eval_q_net(torch.FloatTensor(eval_state).to(device))
                eval_action = dqn.greedy_action(eval_qvals).item()
                eval_next_state, eval_reward, edone,_ = env.step(eval_action)
                eps_reward+=eval_reward
                if edone:
                    break
                eval_state=eval_next_state
            total_reward+=eps_reward

        print("Episode:"+format(episode)+",eval score:"+format(total_reward/10))
        all_rewards.append(total_reward)

print('Complete')
plt.plot(all_rewards)
plt.ylabel('Rewards')
file_name = res_dir + '/DQN_cartpole_rewards.png'
plt.savefig(file_name)