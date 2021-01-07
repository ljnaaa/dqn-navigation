#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import numpy as np
from TD3 import TD3
import time
import rospy
from utils import ReplayBuffer,MultiStepMemory,PER
from PER import Memory
from std_msgs.msg import Float32MultiArray
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Env.environment_stage_2 import Env


def train(state_dim,action_dim):
    ######### Hyperparameters #########
    log_interval = 5           # print avg reward after interval
    gamma = 0.99                # discount for future rewards
    batch_size = 256            # num of transitions sampled from replay buffer
    lr = 1e-4
    exploration_noise = 1.0
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 1000         # max num of episodes
    max_timesteps = 300        # max timesteps in one episode
    load_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),"steer&spd/models16_basic") # save trained models
    directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),"steer&spd/models16") # save trained models
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Make Folder")
    else:
        print("Folder Exists")
    load_filename = "TD3_{}".format("stage2")
    filename = "TD3_{}".format("stage2")
    ###################################
    

    max_action = 1.0
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    policy.load(load_directory,load_filename)
    replay_buffer = ReplayBuffer()
    # replay_buffer = MultiStepMemory(5,0.99)
    # replay_buffer = replay_buffer = PER(10000,5,0.99)
    print("normal")
    # replay_buffer = Memory(5e4,3,0.99)

    env = Env()

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_path = os.path.join(directory,"log.txt")
    log_f = open(log_path,"w+")
    
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        if exploration_noise>0.25:
            exploration_noise = exploration_noise*0.995
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space_shape)
            action = action.clip(-1,1)
            action_msg = Float32MultiArray()
            # take action in env:
            next_state, reward, done, finish = env.step(action)
            
            replay_buffer.add((state, action, reward, next_state, float(finish)))
            # replay_buffer.store((state, action, reward, next_state, float(finish)))
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            action_msg.data = [action[0],action[1],ep_reward,reward]
            pub_get_action.publish(action_msg)
            # if episode is done then update policy:

            if done or t==(max_timesteps-1):
                rospy.loginfo("END AT STEP %d",t)
                result_msg = Float32MultiArray()
                result_msg.data = [ep_reward,episode]
                pub_result.publish(result_msg)
                if len(replay_buffer)>batch_size:
                    policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break
        
        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0
        
        if episode > 100 and episode%10 == 0:
            policy.save(directory, filename)
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

if __name__ == '__main__':
    rospy.init_node("turtlebot3_td3")
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    state_dim = 28
    action_dim = 2    #只控制旋转
    train(state_dim,action_dim)
    
