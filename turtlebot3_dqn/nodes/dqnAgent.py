import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import torch.optim as optim
EPISODES = 3000
import torch.nn.functional as F
import math
# from SumTree import SumTree

# class Memory:  # stored as ( s, a, r, s_ ) in SumTree
#     e = 0.01
#     a = 0.6
#     beta = 0.3
#     beta_increment_per_sampling = 1e-4


#     def __init__(self, capacity):
#         self.tree = SumTree(capacity)
#         self.capacity = capacity
#         self.length = 0
#         self.max_prio = 20

#     def __len__(self):
#         return self.length

#     def _get_priority(self, error):
#         return (np.abs(error) + self.e) ** self.a

#     def add(self, sample):
#         self.length += 1
#         p = self._get_priority(self.max_prio)
#         self.tree.add(p, sample)

#     def sample(self, n):
#         batch = []
#         idxs = []
#         segment = self.tree.total() / n
#         priorities = []

#         self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])


#         for i in range(n):
#             a = segment * i
#             b = segment * (i + 1)

#             s = random.uniform(a, b)
#             (idx, p, data) = self.tree.get(s)
#             priorities.append(p)
#             batch.append(data)
#             idxs.append(idx)

#         return batch, idxs

#     def update(self, idx, error):
#         self.max_prio = max(self.max_prio,max(np.abs(error)))
#         for i, idx in enumerate(idx):
#             p = self._get_priority(error[i])
#             self.tree.update(idx, p)

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    len = 0
    minEle = 2

    def __init__(self, capacity,n,gamma):
        self.tree = SumTree(capacity)
        self.nstep = n
        self.nstep_buffer = deque(maxlen = self.nstep)
        self.gamma = gamma 

    def store(self, transition):
        self.nstep_buffer.append(transition)
        if(len(self.nstep_buffer) == self.nstep):
            state,action = self.nstep_buffer[0][:2]
            rewards,next_state,done = self._cal_nstep_reward()
            transition = (state,action,rewards,next_state,done)
            max_p = np.max(self.tree.tree[-self.tree.capacity:])
            if max_p == 0:
                max_p = self.abs_err_upper
            self.tree.add(max_p, transition)   # set the max p for new p
            self.minEle = min(max_p,self.minEle)
            self.len+=1

    def _cal_nstep_reward(self):
        rewards,next_state,done = self.nstep_buffer[-1][-3:]
        
        for transition in reversed(list(self.nstep_buffer)[:-1]):
            rew,next_s,d = transition[-3:]
            if d:
                rewads = rew
                next_state,done = next_s,d
            else:
                rewards = rew + self.gamma * rewards
        return rewards,next_state,done

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [], np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = self.minEle / self.tree.total_p     # for later calculate ISweight
        if self.len%1000==0:
            print("min_prob:",self.minEle)
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p

            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            
            b_idx[i]= idx
            b_memory.append(data)
        return b_memory, b_idx, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            self.minEle = min(p,self.minEle)

    def __len__(self):
        return self.len

    




class MultiStepMemory:
    def __init__(self,n,gamma,maxLength = 100000):
        self._memory = deque(maxlen = maxLength)
        self.nstep = n
        self.nstep_buffer = deque(maxlen = self.nstep)
        self.gamma = gamma    #reduce index
    
    def store(self,transition):
        self.nstep_buffer.append(transition)
        if(len(self.nstep_buffer) == self.nstep):
            state,action = self.nstep_buffer[0][:2]
            rewards,next_state,done = self._cal_nstep_reward()
            self._memory.append((state,action,rewards,next_state,done))
            
    def sample(self,batch_size):
        return random.sample(self._memory,batch_size)

    def _cal_nstep_reward(self):
        rewards,next_state,done = self.nstep_buffer[-1][-3:]
        
        for transition in reversed(list(self.nstep_buffer)[:-1]):
            rew,next_s,d = transition[-3:]
            if d:
                rewads = rew
                next_state,done = next_s,d
            else:
                rewards = rew + self.gamma * rewards
        return rewards,next_state,done

    def __len__(self):
        return len(self._memory)

class NoisyLinear(nn.Module):
    def __init__(self,in_features,out_features,std_init=0.6):
        super(NoisyLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(self.out_features,self.in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.out_features,self.in_features))

        self.register_buffer("weight_epsilon",torch.FloatTensor(out_features,in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.out_features))
        self.register_buffer("bias_epsilon",torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        
        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self,x):
        return F.linear(x,self.weight_mu + self.weight_sigma*self.weight_epsilon,
        self.bias_mu+self.bias_sigma*self.bias_epsilon)

    def scale_noise(self,size):
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))
        return x.sign().mul(x.abs().sqrt())

    

    




class Network(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        dropout= 0.2
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,self.out_dim)
        )

    def forward(self,x):
        # x = torch.tensor(x).float()
        return self.feature_layer(x.cuda())

class Distributional_Network(nn.Module):
    def __init__(self,in_dim,out_dim,atom_size,vmin,vmax,dropout=0.2):
        super(Distributional_Network,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.atom_size = atom_size
        self.values = torch.linspace(vmin,vmax,atom_size).cuda()

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,self.out_dim*self.atom_size)
        )


    def forward(self,x):
        dist = self.dist(x)
        q = torch.sum(dist*self.values,dim=2)
        return q

    def dist(self,x):
        q_atoms = self.feature_layer(x.cuda()).view(-1,self.out_dim,self.atom_size)
        dist = F.softmax(q_atoms,dim=-1)
        dist = dist.clamp(1e-3)    #limit area of output
        return dist

 
class DuelingNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, dropout= 0.2):
        """Initialization."""
        super(DuelingNetwork, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out1 = nn.Linear(64,1)
        self.out2 = nn.Linear(64,out_dim)


    def forward(self,x):
        # x = torch.tensor(x).float()
        x = x.reshape(-1,self.in_dim)
        x = self.feature_layer(x.cuda())
        y1 = self.out1(x)
        y2 = self.out2(x)
        y2 -= y2.mean(1).reshape(-1,1).expand_as(y2)
        return y1.expand_as(y2) + y2

class NoisyDuelingNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, dropout= 0.2):
        super(NoisyDuelingNetwork,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out1 = NoisyLinear(64,1)
        self.out2 = NoisyLinear(64,out_dim)


    def forward(self,x):
        # x = torch.tensor(x).float()
        x = x.reshape(-1,self.in_dim)
        x = self.feature_layer(x.cuda())
        y1 = self.out1(x)
        y2 = self.out2(x)
        y2 -= y2.mean(1).reshape(-1,1).expand_as(y2)
        return y1.expand_as(y2) + y2

    def reset_noise(self):
        self.out1.reset_noise()
        self.out2.reset_noise()

class RainbowNetwork(nn.Module):
    def __init__(self,in_dim,out_dim,atom_size,vmin,vmax,dropout=0.2):
        super(RainbowNetwork,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.atom_size = atom_size
        self.values = torch.linspace(vmin,vmax,atom_size).cuda()

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 64), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64,32),
            nn.ReLU()
            )


        # self.feature_layer = nn.Sequential(
        #     nn.Linear(in_dim-2, 32), 
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(32,14),
        #     nn.ReLU()
        #     )
        
        # # self.combine_layer = nn.Linear(64,64)
        # self.combine_layer =  nn.Linear(16,64)
        
        self.final_layer = NoisyLinear(64,32)
        # self.out1 = nn.Linear(64,atom_size)
        # self.out2 = nn.Linear(64,atom_size*out_dim)
        self.out1 = NoisyLinear(32,atom_size)
        self.out2 = NoisyLinear(32,atom_size*out_dim)
        
    def forward(self,x):
        x = x.reshape(-1,self.in_dim).cuda()
        # laser,state = torch.split(x,self.in_dim-2,1)
        feature = self.feature_layer(x)
        # feature = torch.cat([feature,state],dim=1)
        # feature = self.combine_layer(feature)
        # nn.ReLU(inplace=True)
        # feature = self.final_layer(feature)
        # nn.ReLU(inplace=True)
        dist,q_atoms1,q_atoms2 = self.dist(feature)
        q = torch.sum(dist * self.values,dim=2)
        return q

        # dist1,dist2 = self.dist(feature)
        # y1 = torch.sum(dist1*self.values,dim=2)
        # y2 = torch.sum(dist2*self.values,dim=2)
        # y2 -= y2.mean(1).reshape(-1,1).expand_as(y2)

        # return y1.expand_as(y2) + y2

    def dist(self,x):
        q_atoms_out1 = self.out1(x).view(-1,1,self.atom_size)

        q_atoms_out2 = self.out2(x).view(-1,self.out_dim,self.atom_size)

        dist = q_atoms_out1 + q_atoms_out2 - q_atoms_out2.mean(1).reshape(-1,1,self.atom_size)
        
        dist = F.softmax(dist,dim=-1)
        dist = dist.clamp(1e-3)    #limit area of output
        return dist,q_atoms_out1,q_atoms_out2 - q_atoms_out2.mean(1).reshape(-1,1,self.atom_size)

    def reset_noise(self):
        self.out1.reset_noise()
        self.out2.reset_noise()
        self.final_layer.reset_noise()

class DQNReinforceAgent(object):
    def __init__(self, state_size, action_size,step=1,PER=False):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_2_')
        self.result = Float32MultiArray()
        self.PER = PER
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.0005
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 16
        self.train_start = 16
        self.dropout = 0.2
        
        if PER:
            self.memory = deque(maxlen=1000000)
        else:
            self.memory = MultiStepMemory(step,0.99)
        # self.memory = PrioritizedReplayBuffer(self.state_size,1000000,self.batch_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Network(self.state_size,self.action_size).to(self.device)
        self.target_model = Network(self.state_size,self.action_size).to(self.device)

        self.updateTargetModel()

        self.lossfn = nn.MSELoss().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)



    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * torch.argmax(next_target)

    def updateTargetModel(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model(torch.tensor(state.reshape(1, len(state))).float())
            self.q_value = q_value.detach().cpu().numpy()
            return np.argmax(self.q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))
        # if not self.PER:
        #     self.memory.store((state, action, reward, next_state, done))
        # else:
        #     target = self.model(torch.FloatTensor(state).cuda()).data
        #     old_val = target[0][action].detach().cpu().data
        #     target_val = self.target_model(torch.FloatTensor(next_state).cuda()).data
        #     if done:
        #         target[0][action] = reward
        #     else:
        #         target[0][action] = reward + self.discount_factor * torch.max(target_val)
        #     error = abs(old_val-target[0][action])
        #     error = error.detach().cpu().numpy()
        #     if error>100:
        #         print("error:",error)
        #     self.memory.add(error,(state, action, reward, next_state, done))

    def modelBP(self,X,Y,indices=None,weights=None):
        if not self.PER:
            self.optimizer.zero_grad()
            loss = self.lossfn(self.model(X),Y)
            loss.backward()
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            pred = self.model(X).max(1)[0]
            pred = pred.reshape(-1,1)
            weights = torch.FloatTensor(weights).cuda()
            loss = weights * (pred - Y.detach()).pow(2) 
            loss = loss.mean()
            loss.backward()

            errors = torch.sum(torch.abs(pred-Y),dim=1)
            self.memory.batch_update(indices,errors.detach().cpu().numpy())
            self.optimizer.step()


    def trainModel(self, target=False):
        if not self.PER:
            mini_batch = self.memory.sample(self.batch_size)
        else:
            mini_batch, idxs,weights = self.memory.sample(self.batch_size)
        # mini_batch,indices,weights = self.memory.sample_batch(0.6)
        # X_batch = np.empty((0, self.state_size), dtype=np.float64)
        # Y_batch = np.empty((0, self.action_size), dtype=np.float64)
        # Y_batch = torch.zeros((0,self.action_size)).to(self.device)
        Y_batch = torch.zeros((0,1)).to(self.device)
        X_batch = torch.zeros((0,self.state_size)).to(self.device)

        for i in range(self.batch_size):
            states = torch.tensor(mini_batch[i][0]).float()
            actions = int(mini_batch[i][1])
            rewards =mini_batch[i][2]
            next_states = torch.tensor(mini_batch[i][3]).float()
            dones = mini_batch[i][4]
            # q_value = self.model(states.reshape(1, len(states)))
            # self.q_value = q_value.detach().cpu().numpy()
            if target:
                next_target = self.target_model(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_states, next_target, dones)
            # Y_sample = q_value.clone().detach()
            # Y_sample[0][actions] = next_q_value
            # Y_sample[0] = Y_sample[0].reshape(-1,self.action_size)
            
            if dones:
                X_batch = torch.cat((X_batch,states.reshape(-1,self.state_size).cuda()))
                Y_batch = torch.cat((Y_batch,torch.tensor(rewards).reshape(-1,1).cuda()), 0)
                # Y_batch = torch.cat((Y_batch,torch.tensor([rewards]*self.action_size).reshape(-1,self.action_size).float().cuda()), 0)
                # print(states[-2:],rewards)
                # X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                # Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)
            else:
                X_batch = torch.cat((X_batch,states.reshape(-1,self.state_size).cuda()))
                Y_batch = torch.cat((Y_batch,next_q_value),0)
                # Y_batch = torch.cat((Y_batch,Y_sample[0].reshape(-1,self.action_size)), 0)
            

        if not self.PER:
            self.modelBP(X_batch,Y_batch)
        else:
            self.modelBP(X_batch,Y_batch,idxs,weights)


class DDQNReinforceAgent(DQNReinforceAgent):
    def getQvalue(self, reward, next_states,next_target, done):
        q_fornext = self.model(next_states)
        max_action_next = torch.argmax(q_fornext,dim=1) 
        
        if done:
            return torch.tensor(reward)
        else:
            target_Q = next_target[:,max_action_next].detach()
            # print(q_fornext.index_select(1,torch.tensor[max_action_next]))
            return reward + self.discount_factor * target_Q

class DuelingDDQN(DDQNReinforceAgent):
    def __init__(self, state_size, action_size,n,PER=False):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/rainbow_PER')
        self.result = Float32MultiArray()
        self.PER = PER
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 16
        self.train_start = 16
        self.dropout = 0.2
        if not self.PER:
            self.memory = MultiStepMemory(n,0.99)
        else:
            self.memory = Memory(100000)
        # self.memory = PrioritizedReplayBuffer(self.state_size,1000000,self.batch_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DuelingNetwork(self.state_size,self.action_size).to(self.device)
        self.target_model = DuelingNetwork(self.state_size,self.action_size).to(self.device)

        self.updateTargetModel()

        self.lossfn = nn.MSELoss().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)      


class DistributionalDDQN(DDQNReinforceAgent):
    def __init__(self, state_size, action_size, step=1, PER=False):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_1_')
        self.result = Float32MultiArray()
        self.PER = PER
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.0005
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 16
        self.train_start = 16
        self.dropout = 0.2
        atom_size = 51
        vmin = 0
        vmax = 100
        if PER:
            self.memory = deque(maxlen=1000000)
        else:
            self.memory = MultiStepMemory(step,0.99)
        # self.memory = PrioritizedReplayBuffer(self.state_size,1000000,self.batch_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Distributional_Network(self.state_size,self.action_size,atom_size,vmin,vmax).to(self.device)
        self.target_model = Distributional_Network(self.state_size,self.action_size,atom_size,vmin,vmax).to(self.device)

        self.updateTargetModel()

        self.lossfn = nn.MSELoss().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)


class NoisyDuelingDDQN(DuelingDDQN):
    def __init__(self, state_size, action_size,n,PER=False):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_1_')
        self.result = Float32MultiArray()
        self.PER = PER
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.0005
        # self.epsilon = 1.0
        # self.epsilon_decay = 0.99
        # self.epsilon_min = 0.05
        self.batch_size = 16
        self.train_start = 16
        self.dropout = 0.2
        if not self.PER:
            self.memory = MultiStepMemory(n,0.99)
        else:
            self.memory = Memory(100000)
        # self.memory = PrioritizedReplayBuffer(self.state_size,1000000,self.batch_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NoisyDuelingNetwork(self.state_size,self.action_size).to(self.device)
        self.target_model = NoisyDuelingNetwork(self.state_size,self.action_size).to(self.device)

        self.updateTargetModel()

        self.lossfn = nn.MSELoss().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)      

    def getAction(self, state):
        q_value = self.model(torch.tensor(state.reshape(1, len(state))).float())
        self.q_value = q_value.detach().cpu().numpy()
        return np.argmax(self.q_value[0])

    def modelBP(self, X, Y, indices=None, weights=None):
        super(NoisyDuelingDDQN,self).modelBP(X, Y, indices=indices, weights=weights)
        # self.target_model.reset_noise()
        # self.model.reset_noise()

class Rainbow(NoisyDuelingDDQN):
    def __init__(self, state_size, action_size,n,PER=False,use_noise=True,epoch=None):
        self.pub_loss = rospy.Publisher('loss', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_2_')
        self.NewLoss = Float32MultiArray()
        self.PER = PER
        self.use_noise = use_noise
        
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 1500
        self.discount_factor = 0.99
        self.learning_rate = 5e-4 
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 32
        self.train_start = 32
        self.dropout = 0.2
        self.lossMemory=[]
        self.current = 0
        atom_size = 51
        vmin = -150
        vmax = 200
        if not self.PER:
            self.memory = MultiStepMemory(n,0.99)
        else:
            self.memory = Memory(20000,n,0.99)
        # self.memory = PrioritizedReplayBuffer(self.state_size,1000000,self.batch_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = RainbowNetwork(self.state_size,self.action_size,atom_size,vmin,vmax).to(self.device)
        self.target_model = RainbowNetwork(self.state_size,self.action_size,atom_size,vmin,vmax).to(self.device)
        if epoch == None:
            self.load_episode = 0
            self.updateTargetModel()
        else:
            self.load_model(epoch)
            self.load_episode = epoch

        self.lossfn = nn.MSELoss().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate) 


    def load_model(self,epoch):
        dir = "/home/cmq/ljn/RL/turtlebot3/src/dqn-navigation/turtlebot3_dqn/save_model"
        model = os.path.join(dir,"stage_1_"+str(epoch)+".pth")
        target_model = os.path.join(dir,"stage_1_"+str(epoch)+"target"+".pth")

        params = torch.load(model)
        self.model.load_state_dict(params)
        params = torch.load(target_model)
        self.target_model.load_state_dict(params)


    def modelBP(self, X, Y, indices=None, weights=None):
        super(Rainbow,self).modelBP(X, Y, indices=indices, weights=weights)
        if self.use_noise:
            self.target_model.reset_noise()
            self.model.reset_noise()

    def getAction(self, state):
        if self.use_noise:
            q_value = self.model(torch.tensor(state.reshape(1, len(state))).float())
            self.q_value = q_value.detach().cpu().numpy()
            return np.argmax(self.q_value[0])
        else:
            if np.random.rand() <= self.epsilon:
                self.q_value = np.zeros(self.action_size)
                return random.randrange(self.action_size)
            else:
                q_value = self.model(torch.tensor(state.reshape(1, len(state))).float())
                self.q_value = q_value.detach().cpu().numpy()
                return np.argmax(self.q_value[0])

    def saveLoss(self):
        pass
        # f = open(self.dirPath+"2.txt",'a')
        # for i in range(self.current,len(self.lossMemory)):
        #     f.writelines(str(self.lossMemory[i])+"\n")
        # f.close()
        # self.current = len(self.lossMemory)
        # self.NewLoss.data = self.lossMemory[self.current:]
        # self.pub_loss.publish(self.NewLoss)
        # self.current = len(self.lossMemory)
                
    def inference(self,state):
        action_q = self.model(torch.FloatTensor(state).cuda())
        action = np.argmax(action_q.cpu().detach().numpy())
        return action