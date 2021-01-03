#!/usr/bin/env python
# -*- coding:utf-8 -*-
import rospy
import os
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
# from src.turtlebot3_dqn.environment_stage_1_new import Env
from src.turtlebot3_dqn.environment_stage_2 import Env
from dqnAgent import *
import torch
EPISODES = 3000


def UseEpsilon(Network):
    rospy.init_node('dqn_node')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    env = Env(action_size)
    agent = Network
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done,finish = env.step(action)
            agent.appendMemory(state, action, reward, next_state, finish)
            # agent.memory.store(state,action,reward,next_state,done)
            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

    
            if e % 20 == 0 and t==0:
                torch.save(agent.model.state_dict(),agent.dirPath + str(e) + '.pth')
                torch.save(agent.target_model.state_dict(),agent.dirPath + str(e) + 'target.pth')
                print("epsilon:",agent.epsilon)
                # agent.model.save(agent.dirPath + str(e) + '.h5')
                # with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                #     json.dump(param_dictionary, outfile)

            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                agent.saveLoss()
                result.data = [score, np.max(agent.q_value),e]
                pub_result.publish(result)
                # agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay


def UseNoise(Network):
    rospy.init_node('turtlebot3_dqn_stage_1')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    # state_size = 26
    # action_size = 5

    env = Env(action_size)

    agent = Network
    scores, episodes = [], []
    if agent.load_episode:
        global_step = agent.target_update
    else:
        global_step = 0
    
    start_time = time.time()
    
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)
            next_state, reward, done, finish= env.step(action)
            # print(next_state, reward, collision, finish)
            agent.appendMemory(state, action, reward, next_state, finish)
            # agent.memory.store(state,action,reward,next_state,done)
            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)
            train = time.time()
            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)
            
            if e % 20 == 0 and t==0:
                torch.save(agent.model.state_dict(),agent.dirPath + str(e) + '.pth')
                torch.save(agent.target_model.state_dict(),agent.dirPath + str(e) + 'target.pth')
                # agent.model.save(agent.dirPath + str(e) + '.h5')
                # with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                #     json.dump(param_dictionary, outfile)

            if t >= 500:
                done = True
                rospy.loginfo("Time out!!")
                agent.saveLoss()



            if done:
                state = env.reset()
                rospy.loginfo('step: %d',t)
                result.data = [score, np.max(agent.q_value),t]
                pub_result.publish(result)
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d time: %d:%02d:%02d',
                              e, score, len(agent.memory), h, m, s)
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")
                agent.updateTargetModel()


def Inference(agent):
    rospy.init_node("turtlebot3_dqn_stage_1")
    env = Env(action_size)
    rate = rospy.Rate(10)
    env.reset()
    while(not rospy.is_shutdown()):
        state = env.Humanstep()
        agent.inference(state)
        rate.sleep()



if __name__ == '__main__':
    state_size = 26
    action_size = 5
    agent = Rainbow(state_size,action_size,3,True,True)
    UseNoise(agent)
    # Inference(agent)