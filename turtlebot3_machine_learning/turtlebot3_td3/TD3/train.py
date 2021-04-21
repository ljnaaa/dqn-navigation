#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import numpy as np
from TD3 import TD3
import time
import rospy
from utils import ReplayBuffer, MultiStepMemory, PER
from PER import Memory
from std_msgs.msg import Float32MultiArray
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from Env.environment_stage_2 import Env
from Env.environment_icra import Env


class importDataset(object):
    def __init__(self, state_dim, action_dim):
        self.datasetPath = "/home/cmq/ljn/RL/turtlebot3/src/dqn-navigation/turtlebot3_machine_learning/turtlebot3_machine_learning/scripts/buffer.npy"
        self.dataset = np.load(self.datasetPath, allow_pickle=True)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def transfer(self):
        self.buffer = []
        for data in self.dataset:
            state = data[0]
            action = data[1]
            reward = data[2]
            next_state = data[3]
            finish = float(data[4])
            assert len(state) == state_dim
            assert(len(action)) == action_dim
            assert len(next_state) == state_dim
            self.buffer.append((state, action, reward, next_state, finish))
        return self.buffer


def train(state_dim, action_dim):
    ######### Hyperparameters #########
    log_interval = 5           # print avg reward after interval
    gamma = 0.99                # discount for future rewards
    batch_size = 256            # num of transitions sampled from replay buffer
    lr = 1e-4
    exploration_noise = 0.8
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 1.0         # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 200         # max num of episodes
    max_timesteps = 500        # max timesteps in one episode
    pretrain_times = 200
    warmup_epoch = 0
    # load_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),"steer&spd/model302_sl") # save trained models
    directory = os.path.join(os.path.dirname(os.path.abspath(
        os.path.dirname(__file__))), "steer&spd/model309")  # save trained models
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Make Folder")
    else:
        print("Folder Exists")
    # load_filename = "TD3_{}".format("stage2")
    filename = "TD3_{}".format("stage2")
    ###################################

    max_action = 1.0

    policy = TD3(lr, state_dim, action_dim, max_action)
    # policy.load(load_directory,load_filename)
    # print("load "+load_directory)
    replay_buffer = ReplayBuffer()
    replay_buffer.importDataset(importDataset(state_dim,action_dim).transfer())
    # replay_buffer = MultiStepMemory(5,0.99)
    # replay_buffer = replay_buffer = PER(10000,5,0.99)
    print("normal Path reward")
    rospy.loginfo("Pretrai n start")
    print(len(replay_buffer))
    # replay_buffer = Memory(5e4,3,0.99)
    for i in range(pretrain_times):
        if(i % 50 == 0):
            rospy.loginfo("finish batch :%d", i)
        policy.actor_SL(replay_buffer, 100, batch_size)

    for i in range(pretrain_times):
        if(i % 50 == 0):
            rospy.loginfo("finish batch :%d", i)
        policy.critic_SL(replay_buffer, 100, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
    rospy.loginfo("Pretrain finish")

    env = Env()

    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_path = os.path.join(directory, "log.txt")
    log_f = open(log_path, "a+")

    # # training procedure:
    # for episode in range(1, max_episodes+1):
    #     state = env.reset()
    #     if exploration_noise>0.25:
    #         exploration_noise = exploration_noise*0.995
    #     for t in range(max_timesteps):
    #         # select action and add exploration noise:
    #         action = policy.select_action(state)
    #         action = action + np.random.normal(0, exploration_noise, size=env.action_space_shape)
    #         action = action.clip(-1,1)
    #         action_msg = Float32MultiArray()
    #         # take action in env:
    #         next_state, reward, done, finish = env.step(action)
    #         last_action = action

    #         replay_buffer.add((state, action, reward, next_state, float(finish)))
    #         # replay_buffer.store((state, action, reward, next_state, float(finish)))
    #         state = next_state

    #         avg_reward += reward
    #         ep_reward += reward
    #         action_msg.data = [action[0],action[1],ep_reward,reward]
    #         pub_get_action.publish(action_msg)
    #         # if episode is done then update policy:

    #         if done or t==(max_timesteps-1):
    #             rospy.loginfo("END AT STEP %d",t)
    #             result_msg = Float32MultiArray()
    #             result_msg.data = [ep_reward,episode]
    #             pub_result.publish(result_msg)
    #             if len(replay_buffer)>batch_size:
    #                 policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
    #             break

    #     # logging updates:
    #     log_f.write('{},{}\n'.format(episode, ep_reward))
    #     log_f.flush()
    #     ep_reward = 0

    #     if episode > 100 and episode%10 == 0:
    #         policy.save(directory, filename)

    #     # print avg reward every log interval:
    #     if episode % log_interval == 0:
    #         avg_reward = int(avg_reward / log_interval)
    #         print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
    #         avg_reward = 0

# step 2
    print("STEP 2 START")
    exploration_noise = 0.6
    max_episodes = 1000
    for episode in range(1, max_episodes+1):
        state = env.reset()
        if exploration_noise > 0.05:
            exploration_noise = exploration_noise*0.995
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + \
                np.random.normal(0, exploration_noise,
                                 size=env.action_space_shape)
            action = action.clip(-1, 1)
            action_msg = Float32MultiArray()
            # take action in env:
            # next_state, reward, done, finish = env.step(action,MoveReward=True)
            next_state, reward, done, finish = env.step(action)
            last_action = action

            replay_buffer.add(
                (state, action, reward, next_state, float(finish)))
            # if(len(replay_buffer)>10000):
            #     buffer = np.array(replay_buffer.buffer)
            #     np.save("buffer",buffer)
            #     print("save buffer")
            # return
            # replay_buffer.store((state, action, reward, next_state, float(finish)))
            state = next_state

            avg_reward += reward
            ep_reward += reward
            action_msg.data = [action[0], action[1], ep_reward, reward]
            pub_get_action.publish(action_msg)
            # if episode is done then update policy:

            if done or t == (max_timesteps-1):
                rospy.loginfo("END AT STEP %d", t)
                result_msg = Float32MultiArray()
                result_msg.data = [ep_reward, episode]
                pub_result.publish(result_msg)
                if len(replay_buffer) > batch_size/2:
                    policy.update(replay_buffer, t, batch_size, gamma,
                                  polyak, policy_noise, noise_clip, policy_delay)
                warmup = 100
                if len(replay_buffer)>batch_size/2:
                    if(episode<warmup_epoch):
                        warmup = 2
                        rospy.loginfo("warm up")
                    elif (episode==warmup_epoch):
                        replay_buffer.move_data()
                        rospy.loginfo("data move")
                    t = max(t,100)
                    policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay,warmup)
                break

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0

        if episode > 100 and episode % 10 == 0:
            policy.save(directory, filename)

        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0


if __name__ == '__main__':
    rospy.init_node("turtlebot3_td3")
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher(
        'get_action', Float32MultiArray, queue_size=5)
    state_dim = 64
    action_dim = 2  # 只控制旋转
    train(state_dim, action_dim)
