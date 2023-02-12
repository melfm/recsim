import numpy as np
import os
import matplotlib.pyplot as plt

import gym
import torch


gamma=0.9
episilon=0.9
lr=0.001
target_update_iter=100
log_internval=100
train_episodes = 5000
terminal_step = 300

env=gym.make('CartPole-v0')
device="cuda"
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]
hidden=256

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

    def learn(self,s,a,s_,r,done):
        eval_q=self.net(torch.Tensor(s))[a]
        target_q=self.target_net(torch.FloatTensor(s_))
        target_a=self.choose_action(target_q)
        target_q=target_q[target_a]
        if not done:
            y=gamma*target_q+r
        else:
            y=r
        loss=(y-eval_q)**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iter_num+=1
        if self.iter_num%10==0:
            self.target_net.load_state_dict(self.net.state_dict())
        return target_a, loss.item()

    def greedy_action(self,qs):
        return torch.argmax(qs)

    def random_action(self):
        return np.random.randint(0,n_action)

    def choose_action(self,qs):
        if np.random.rand()>episilon:
            return self.random_action()
        else:
            return self.greedy_action(qs).tolist()

sarsa=Sarsa()
all_rewards = []
all_losses = []

for episode in range(train_episodes):
    state = env.reset()
    train_step = 0
    reward = 0.0
    qvals = sarsa.net(torch.Tensor(state))
    action = sarsa.choose_action(qvals)
    while(train_step<terminal_step):
        train_step+=1
        next_state,reward,done,_ = env.step(action)
        action, loss = sarsa.learn(state, action, next_state, reward, done)
        state = next_state
        all_losses.append(loss)
        if done:
            break
    if episode%log_internval==0:
        # Evaluation
        total_reward=0.0
        for i in range(10):
            eval_state = env.reset()
            eval_reward = 0.0
            eps_reward = 0.0
            eval_step = 0
            while(eval_step<terminal_step):
                eval_step += 1
                eval_qvals = sarsa.net(torch.Tensor(eval_state))
                action = sarsa.greedy_action(eval_qvals)
                eval_next_state, eval_reward, edone,_=env.step(action.tolist())
                eps_reward+=eval_reward
                if edone:
                    edone = False
                    break
                eval_state=eval_next_state
            total_reward+=eps_reward
        print("Episode:"+format(episode)+", eval reward:"+format(total_reward/10))
        all_rewards.append(total_reward)


print('Complete')
plt.plot(all_rewards)
plt.ylabel('Rewards')
file_name = res_dir + '/Sarsa_cartpole_rewards.png'
plt.savefig(file_name)

plt.clf()

plt.plot(all_losses)
plt.ylabel('TD_Error')
file_name = res_dir + '/Sarsa_cartpole_losses.png'
plt.savefig(file_name)


