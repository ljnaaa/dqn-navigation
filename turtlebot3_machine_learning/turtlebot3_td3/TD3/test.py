#!/usr/bin/env python
# -*- coding:utf-8 -*-

from rosgraph import network
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


    


class NetworkTest(object):
    def __init__(self,NetworkList,totalGoal=10):
        self.NetworkList = NetworkList
        self.totalStep = np.zeros(len(NetworkList))
        self.totalSucc = np.zeros(len(NetworkList))
        self.goalList = []
        self.totalGoal = totalGoal
        self.env = Env()

    def test(self):
        for i in range(len(self.NetworkList)):
            self.inference(i)
        for i in range(len(self.NetworkList)):
            rospy.loginfo("Network {} is test completed,total Step is {},SuccRate is {}".format(i,self.totalStep[i],(self.totalSucc[i]+1.0)/self.totalGoal))
        

    def inference(self,i):
        max_timesteps = 300
        if len(self.goalList) == self.totalGoal:
            state = self.env.reset(self.goalList[0][0],self.goalList[0][1])
            # training procedure:
            for episode in range(1, self.totalGoal):
                self.goalList.append(self.env.getGoal())
                for t in range(max_timesteps):
                    # select action and add exploration noise:
                    action = self.NetworkList[i].select_action(state)
                    # take action in env:
                    next_state, reward, done, finish = self.env.step(action,self.goalList[episode][0],self.goalList[episode][1])
                    state = next_state

                    if finish:
                        # rospy.loginfo("Mission {} Complete use step {}".format(episode,t))
                        self.totalSucc[i] += 1
                        self.totalStep[i] += t
                        break

                    if done or t==(max_timesteps-1):
                        state = self.env.reset(self.goalList[episode][0],self.goalList[episode][1])
                        self.totalStep[i] += t
                        break
        else:
            state = self.env.reset()
            # training procedure:
            for episode in range(1, self.totalGoal):
                self.goalList.append(self.env.getGoal())
                for t in range(max_timesteps):
                    # select action and add exploration noise:
                    action = self.NetworkList[i].select_action(state)
                    # take action in env:
                    next_state, reward, done, finish = self.env.step(action)
                    state = next_state

                    if finish:
                        self.totalSucc[i] += 1
                        self.totalStep[i] += t
                        break

                    if done or t==(max_timesteps-1):
                        state = self.env.reset()
                        self.totalStep[i] += max_timesteps
                        break
            self.goalList.append(self.env.getGoal())
        
        rospy.loginfo("Network {} is test completed,total Step is {}".format(i,self.totalStep[i]))

def loadNetwork(pathList,state_dim,action_dim):
    lr = 0
    max_action = 1.0
    networklist = []
    for path in pathList:
        load_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),path)
        load_filename = "TD3_{}".format("stage2")
        policy = TD3(lr, state_dim, action_dim, max_action)
        policy.load(load_directory,load_filename)
        networklist.append(policy)
    return networklist




if __name__ == '__main__':
    rospy.init_node("turtlebot3_td3")
    state_dim = 28
    action_dim = 2    #只控制旋转
    pathlist = ["steer&spd/models111","steer&spd/models13"]
    networklist = loadNetwork(pathlist,state_dim,action_dim)
    NetworkTest(networklist,100).test()
    