import gym
import torch as th
import numpy as np 

batch_size=10
lr=0.001
episilon=0.9
replay_memory_size=10000
gamma=0.9
target_update_iter=100
log_internval=10
env=gym.make('CartPole-v0')
device="cuda"
env=env.unwrapped
n_action=env.action_space.n
n_state=env.observation_space.shape[0]
hidden1=256
hidden2=256

class net(th.nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1=th.nn.Linear(n_state,hidden1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2=th.nn.Linear(hidden1,hidden2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.V=th.nn.Linear(hidden2,1)
        self.V.weight.data.normal_(0,0.1)
        self.A=th.nn.Linear(hidden2,n_action)
        self.A.weight.data.normal_(0,0.1)
    

    def forward(self,x):
        x=self.fc1(x)
        x=th.nn.functional.relu(x)
        x=self.fc2(x)
        x=th.nn.functional.relu(x)
        v=self.V(x)
        a=self.A(x)
        if len(a.shape)==1:
            a-=th.mean(a,dim=0)
        else:   #for batch
            a-=th.mean(a,dim=-1).unsqueeze(1).repeat(1,2)

        if len(a.shape)==1:
            out=a+v.squeeze()
        else:   #for batch
            out=a+v.repeat(1,2)
        return out

class replay_memory():
    def __init__(self):
        self.memory_size=replay_memory_size
        self.memory=np.array([])
        self.cur=0
        self.new=0
    def size(self):
        return self.memory.shape[0]
    #[s,a,r,s_,done]
    def store_transition(self,trans):
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
    
class DQN(object):
    def __init__(self):
        self.eval_q_net,self.target_q_net=net().to(device),net().to(device)
        self.replay_mem=replay_memory()
        self.iter_num=0
        self.optimizer=th.optim.Adam(self.eval_q_net.parameters(),lr=lr)
        self.loss=th.nn.MSELoss().to(device)
    def choose_action(self,qs):
        if np.random.uniform()<episilon:
            return th.argmax(qs).tolist()
        else:
            return np.random.randint(0,n_action)
    def greedy_action(self,qs):
        return th.argmax(qs)
    def learn(self):
        if(self.iter_num%target_update_iter==0):
            self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
        self.iter_num+=1

        batch=self.replay_mem.sample()
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_a=th.LongTensor(batch[:,1].astype(int).tolist()).to(device)
        b_r=th.FloatTensor(batch[:,2].tolist()).to(device)
        b_s_=th.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=th.FloatTensor(batch[:,4].tolist()).to(device)
        q_target=th.zeros((batch_size,1)).to(device)
        q_eval=self.eval_q_net(b_s)
        q_eval=th.gather(q_eval,dim=1,index=th.unsqueeze(b_a,1))
        q_next=self.target_q_net(b_s_).detach()
        for i in range(b_d.shape[0]):
            if(int(b_d[i].tolist()[0])==0):
                q_target[i]=b_r[i]+gamma*th.unsqueeze(th.max(q_next[i],0)[0],0)
            else:
                q_target[i]=b_r[i]
        td_error=self.loss(q_eval,q_target)

        self.optimizer.zero_grad()
        td_error.backward()
        self.optimizer.step()

dqn=DQN()

for episode in range(10000):
    s=env.reset()
    t=0
    r=0.0
    while(t<300):
        t+=1
        qs=dqn.eval_q_net(th.FloatTensor(s).to(device))
        a=dqn.choose_action(qs)
        s_,r,done,_=env.step(a)
        transition=[s.tolist(),a,[r],s_.tolist(),[done]]
        dqn.replay_mem.store_transition(transition)
        s=s_
        if dqn.replay_mem.size()>batch_size:
            dqn.learn()
        if done:
            break
    if episode%log_internval==0: #test
        total_reward=0.0
        for i in range(10):
            t_s=env.reset()
            t_r=0.0
            tr=0.0
            time=0
            while(time<300):
                time+=1
                t_qs=dqn.eval_q_net(th.FloatTensor(t_s).to(device))
                t_a=dqn.greedy_action(t_qs).item()
                ts_,tr,tdone,_=env.step(t_a)
                t_r+=tr
                if tdone:
                    break
                t_s=ts_
            total_reward+=t_r
        print("episode:"+format(episode)+",test score:"+format(total_reward/10))




