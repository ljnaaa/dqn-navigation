#!/usr/bin/env python
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

from numpy.lib.function_base import delete
import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn
import time

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
        self.sub_scan = rospy.Subscriber('scan',LaserScan,self.getScan)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.action_space_shape = 2

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.lastDis = goal_distance
        return goal_distance

    def getScan(self,scan):
        self.scan = scan

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)
        self.vx = odom.twist.twist.linear.x
        self.wz = odom.twist.twist.angular.z

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.2
        done = False
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
            done = True
            finish = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        if current_distance < 0.2:
            self.get_goalbox = True
            finish = True
        # return scan_range + [heading, current_distance], done, finish
        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done, finish

        # return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle,self.vx,self.wz], done, finish

    def setReward(self, state, done, action,goal_x=0,goal_y=0):
        current_distance = state[-3]
        heading = state[-4]
        angle = heading+action[0]*pi/8 +pi/2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * (angle) % (2 * math.pi) / math.pi)[0])   # tr->1 when angle->0 tr->-1 when angle->180
        move_dis = (self.lastDis - current_distance ) * 150
        self.lastDis = current_distance
        # reward = 0
        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = ((round(tr*5, 2)) * distance_rate)*0.2 - 0.2*abs(action[0])
        # reward = move_dis
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
            # rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            if goal_x == 0 and goal_y == 0:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            else:
                self.goal_x, self.goal_y = self.respawn_goal.Setgoal(goal_x,goal_y)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def setMoveReward(self, state, done, action,goal_x=0,goal_y=0):
        current_distance = state[-3]
        heading = state[-4]
        angle = heading+action[0]*pi/8 +pi/2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * (angle) % (2 * math.pi) / math.pi)[0])   # tr->1 when angle->0 tr->-1 when angle->180
        move_dis = (self.lastDis - current_distance ) * 20
        self.lastDis = current_distance
        # reward = 0
        distance_rate = 2 ** (current_distance / self.goal_distance)
        # reward = ((round(tr*5, 2)) * distance_rate)
        reward = move_dis - 0.2*abs(action[0])
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
            # rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            if goal_x == 0 and goal_y == 0:
                self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            else:
                self.goal_x, self.goal_y = self.respawn_goal.Setgoal(goal_x,goal_y)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action,goal_x=0,goal_y=0,MoveReward=False):
        max_angle_vel = 2
        max_linear_spd = 0.3
        vel_cmd = Twist()
        # vel_cmd.linear.x = 0.2
        vel_cmd.linear.x = action[1] * max_linear_spd/2 + max_linear_spd/2
        vel_cmd.angular.z = action[0] * max_angle_vel
        self.pub_cmd_vel.publish(vel_cmd)
        # max_angular_acc = 5.0
        # max_vel_acc = 5.0
        # vel_cmd = Twist()
        # vel_cmd.linear.x = action[1] * max_vel_acc + self.vx
        # vel_cmd.angular.z = action[0] * max_angular_acc +self.wz
        # vel_cmd.linear.x = max(0,min(vel_cmd.linear.x,0.3))
        # vel_cmd.angular.z = max(-2,min(vel_cmd.angular.z,2))
        # self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        state, done, finish = self.getState(self.scan)
        if not MoveReward:
            reward = self.setReward(state, done, action,goal_x,goal_y)
        else:
            reward = self.setMoveReward(state, done, action,goal_x,goal_y)
        return np.asarray(state), reward, done, finish

    def Humanstep(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done, finish = self.getState(data)
        if done:
            rospy.loginfo("Collision!!")
            reward = -150
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return np.asarray(state)

    def reset(self,goal_x=0,goal_y=0):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

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
        state, done, finish = self.getState(data)
        return np.asarray(state)

    
    def getGoal(self):
        return (self.goal_x,self.goal_y)