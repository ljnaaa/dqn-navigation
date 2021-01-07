#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import numpy as np
from TD3 import TD3
import time
import rospy
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from Env.environment_stage_2 import Env
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


    



def train(state_dim,action_dim):
    ######### Hyperparameters #########
    lr = 1e-4
    log_interval = 1
    max_episodes = 1000         # max num of episodes
    max_timesteps = 3000       # max timesteps in one episode
    load_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),"steer&spd/models16") # save trained models
    load_filename = "TD3_{}".format("stage2")
    ###################################
    

    max_action = 1.0
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    policy.load(load_directory,load_filename)

    env = Env()

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    state = env.reset()
    # training procedure:
    for episode in range(1, max_episodes+1):
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            # take action in env:
            next_state, reward, done, finish = env.step(action)
            action_msg = Float32MultiArray()
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            action_msg.data = [action[0],action[1],ep_reward,reward]
            pub_get_action.publish(action_msg)
            # if episode is done then update policy:

            if finish:
                rospy.loginfo("Mission {} Complete use step {}".format(episode,t))
                break

            if done or t==(max_timesteps-1):
                state = env.reset()
                break

        
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
    