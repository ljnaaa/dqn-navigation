#!/usr/bin/env python
# -*- coding:utf-8 -*-
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

from matplotlib.pyplot import angle_spectrum, grid, isinteractive
import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry,OccupancyGrid
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn
import random
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
import os
import time



class MapInfo(object):
    def __init__(self,map):
        self.map = map

    def WorldToGrid(self,world_x,world_y):
        grid_x = self.map.info.origin.position.x + world_x / self.map.info.resolution
        grid_y = self.map.info.origin.position.y + world_y / self.map.info.resolution
        return int(grid_x),int(grid_y)

    def GetIndex(self,x,y):
        return x + y * self.map.info.width

    def Occupied(self,x,y):
        grid_x,grid_y = self.WorldToGrid(x,y)
        index = self.GetIndex(grid_x,grid_y)
        if self.map.data[index]:
            return True
        else:
            return False

class RespawnNew(Respawn):
    def __init__(self,mapinfo):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('turtlebot3_machine_learning/turtlebot3_machine_learning/scripts',
                                                'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        # self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = 0.5
        self.init_goal_y = 1.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0
        self.map = mapinfo

    def SetRange(self,xmin,xmax,ymin,ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def CreateTarget(self):
        find = False
        while not find:
            goal_x = random.uniform(self.xmin,self.xmax)
            goal_y = random.uniform(self.ymin,self.ymax)
            if not self.map.Occupied(goal_x,goal_y):
                find = True
                return goal_x,goal_y


    def getPosition(self,position_check=True,delete=False):
        if delete:
            self.deleteModel()
        
        goal_x,goal_y = self.CreateTarget()
        self.goal_position.position.x = goal_x
        self.goal_position.position.y = goal_y
        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y


        




class Env():
    def __init__(self):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.map_sub = rospy.Subscriber('move_base/global_costmap/costmap',OccupancyGrid,self.mapCb)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        # self.respawn_goal = Respawn()
        rospy.wait_for_message("move_base/global_costmap/costmap",OccupancyGrid)
        
        self.respawn_goal = RespawnNew(self.map)
        self.respawn_goal.SetRange(0.5,8,0.5,4.4)
        self.action_space_shape = 1
        


    def mapCb(self,map):
        if hasattr(self,"map"):
            pass
        else:
            self.map = MapInfo(map)

    

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.lastDis = goal_distance
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi


        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.2
        collision = False
        finish = False
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            collision = True
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        if current_distance < 0.15:
            finish = True
            self.get_goalbox = True
        # return scan_range + [heading, current_distance], done, finish
        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], collision, finish

    def setReward(self, state, done, action,goal_x=0,goal_y=0):
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]
        angle = heading+action[0]*pi/8 +pi/2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * (angle) % (2 * math.pi) / math.pi)[0])   # tr->1 when angle->0 tr->-1 when angle->180
        move_dis = (self.lastDis - current_distance ) * 100
        self.lastDis = current_distance
        reward = 0
        # distance_rate = 2 ** (current_distance / self.goal_distance)
        # reward = ((round(tr*5, 2)) * distance_rate)
        # reward = move_dis - abs(action[0]) * 1 #角度变化惩罚

        reward = move_dis
        if obstacle_min_range < 0.25:
            reward = -30
        if done:
            rospy.loginfo("Collision!!")
            reward = -150
            self.pub_cmd_vel.publish(Twist())
            if goal_x == 0 and goal_y == 0:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            else:
                self.goal_x, self.goal_y = self.respawn_goal.Setgoal(goal_x,goal_y)
            self.goal_distance = self.getGoalDistace()

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            if goal_x == 0 and goal_y == 0:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            else:
                self.goal_x, self.goal_y = self.respawn_goal.Setgoal(goal_x,goal_y)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):
        max_angle_vel = 2
        max_linear_spd = 0.3
        vel_cmd = Twist()
        vel_cmd.linear.x = action[1] * max_linear_spd/2 + max_linear_spd/2
        vel_cmd.angular.z = action[0] * max_angle_vel
        self.pub_cmd_vel.publish(vel_cmd)

        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done, finish = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done, finish

    def reset(self,goal_x=0,goal_y=0):
        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_world service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
        else:
            if goal_x==0 and goal_y==0:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True,delete=True)
            else:
                self.goal_x, self.goal_y = self.respawn_goal.Setgoal(goal_x,goal_y)

        self.goal_distance = self.getGoalDistace()
        state, done,_ = self.getState(data)

        return np.asarray(state)
