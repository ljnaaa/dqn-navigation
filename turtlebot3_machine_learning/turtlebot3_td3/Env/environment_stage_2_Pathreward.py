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

from shutil import move
from numpy.lib.function_base import delete
import rospy
import numpy as np
import math
from math import floor, pi
from geometry_msgs.msg import Twist, Point, Pose,PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry,Path
from nav_msgs.srv import GetPlan
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn
import tf2_ros
import tf


class Env():
    def __init__(self):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pose = PoseStamped()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.action_space_shape = 1
        self.listener=tf.TransformListener()
        self.listener.waitForTransform('map', 'odom', rospy.Time(0), rospy.Duration(5))
        self.path = Path()

        # self.tf_buffer = tf2_ros.Buffer()


    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.getPath()
        self.lastDis = goal_distance
        return goal_distance

    def PathLimit(self,path,length=100):    #limit the length of the path
        if len(path)<=100:
            self.path = path
        else:
            interval = float(len(path))/length
            self.path = []
            for i in range(length):
                index = int(floor(interval * i))
                self.path.append(path[index])

    def CalDis(self):
        min_dis = 999.0
        for point in self.path:
            dis = (self.position.x - point.pose.position.x)**2 + (self.position.y - point.pose.position.y)**2
            dis = math.sqrt(dis)
            min_dis = min(dis,min_dis)
        return min_dis

    def getPath(self):
        goal = PoseStamped()
        goal.header = self.pose.header
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = self.goal_x
        goal.pose.position.y = self.goal_y
        getPathSrv = rospy.ServiceProxy('/move_base/make_plan',GetPlan)
        # self.listener.waitForTransform('map', 'odom', rospy.Time(0), rospy.Duration(5))
        # goal_map = self.listener.transformPose('map',goal)
        # start_map = self.listener.transformPose('map',self.pose)
        goal_map = goal
        goal_map.header.frame_id = "map"
        start_map = self.pose
        start_map.header.frame_id = "map"
        try:
            path = getPathSrv(start_map,goal_map,0.4)
            self.PathLimit(path.plan.poses)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))

    def getOdometry(self, odom):
        self.odom = odom
        self.position = odom.pose.pose.position
        self.pose.header = odom.header
        self.pose.pose = odom.pose.pose
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

    def setReward(self, state, done, action):
        
        current_distance = state[-3]
        heading = state[-4]
        angle = heading+action[0]*pi/8 +pi/2
        tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * (angle) % (2 * math.pi) / math.pi)[0])   # tr->1 when angle->0 tr->-1 when angle->180
        move_dis = (self.lastDis - current_distance ) * 150
        self.lastDis = current_distance
        # reward = 0
        distance_rate = 2 ** (current_distance / self.goal_distance)
        path_dis = self.CalDis()
        path_dis = min(0.5,path_dis)
        path_dis_reward = 1-path_dis*3/0.5
        path_dis_reward = max(-0.5,path_dis_reward)
        move_dis = min(10,move_dis)
        # reward = ((round(tr*5, 2)) * distance_rate)
        reward = move_dis+path_dis_reward
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

        return reward

    def step(self, action):
        max_angle_vel = 2
        max_linear_spd = 0.3
        vel_cmd = Twist()
        vel_cmd.linear.x = action[1] * max_linear_spd/2 + max_linear_spd/2
        vel_cmd.angular.z = action[0] * max_angle_vel
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

    def reset(self):
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
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True,delete=True)

        

        self.goal_distance = self.getGoalDistace()
        state, done, finish = self.getState(data)
        return np.asarray(state)