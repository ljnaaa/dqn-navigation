import numpy as np
import random
from collections import deque
import time

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.size = 0

class MultiStepMemory:
    def __init__(self,n,gamma,maxLength = 100000):
        self._memory = deque(maxlen = maxLength)
        self.nstep = n
        self.nstep_buffer = deque(maxlen = self.nstep)
        self.gamma = gamma    #reduce index
    
    def add(self,transition):
        self.nstep_buffer.append(transition)
        if(len(self.nstep_buffer) == self.nstep):
            state,action = self.nstep_buffer[0][:2]
            rewards,next_state,done = self._cal_nstep_reward()
            self._memory.append((state,action,rewards,next_state,done))
            
    def sample(self,batch_size):
        if len(self._memory)>batch_size:    
            batch = random.sample(self._memory,batch_size)
        else:
            batch = random.sample(self._memory,len(self._memory))
            batch += random.sample(self._memory,batch_size - len(self._memory))
        state, action, reward, next_state, done = [],[],[],[],[]
        for trans in batch:
            state.append(trans[0])
            action.append(trans[1])
            reward.append(trans[2])
            next_state.append(trans[3])
            done.append(trans[4])
        return state, action, reward, next_state, done

    def _cal_nstep_reward(self):
        rewards,next_state,done = self.nstep_buffer[-1][-3:]
        for transition in reversed(list(self.nstep_buffer)[:-1]):
            rew,next_s,d = transition[-3:]
            if d:
                rewards = rew
                next_state,done = next_s,d
            else:
                rewards = rew + self.gamma * rewards
        return rewards,next_state,done

    def __len__(self):
        return len(self._memory)


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


class PER(object):  # stored as ( s, a, r, s_ ) in SumTree
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

    def add(self, transition):
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
                rewards = rew
                next_state,done = next_s,d
            else:
                rewards = rew + self.gamma * rewards
        return rewards,next_state,done

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [], np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = self.minEle / self.tree.total_p     # for later calculate ISweight
        state,action,rewards,next_state,done = [],[],[],[],[]
        if self.len%1000==0:
            print("min_prob:",self.minEle)
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            state.append(data[0])
            action.append(data[1])
            rewards.append(data[2])
            next_state.append(data[3])
            done.append(data[4])
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            
            b_idx[i]= idx
            b_memory.append(data)
        return state,action,rewards,next_state,done, b_idx, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            self.minEle = min(p,self.minEle)

    def __len__(self):
        return self.len
