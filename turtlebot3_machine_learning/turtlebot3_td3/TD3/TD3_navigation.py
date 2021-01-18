#!/usr/bin/env python
# -*- coding:utf-8 -*-

from numpy.lib.index_tricks import _fill_diagonal_dispatcher
from TD3 import TD3
import rospy
import sys
import os
import math
import numpy as np
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped,Point,Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import time



class stage(object):
    def __init__(self):
        self.laserUpdate = False
        self.OdomUpdate = False
        self.InMission = False
        self.scan = LaserScan()
        self.goal = PoseStamped()
        self.start = PoseStamped()
        self.heading = 0.0
        self.position = Point()


    def UpdateScan(self,scan):
        if self.InMission:
            self.scan = scan
            if not self.laserUpdate:
                self.laserUpdate = True

    def UpdateGoal(self,goal):
        self.goal = goal
        if not self.InMission:
            self.InMission = True
            self.laserUpdate = False
            self.OdomUpdate = False
        
    def UpdateOdometry(self,odom):
        if self.InMission:
            self.position = odom.pose.pose.position
            orientation = odom.pose.pose.orientation
            orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
            _, _, yaw = euler_from_quaternion(orientation_list)

            goal_angle = math.atan2(self.goal.pose.position.y - self.position.y, self.goal.pose.position.x - self.position.x)

            heading = goal_angle - yaw
            if heading > math.pi:
                heading -= 2 * math.pi

            elif heading < -math.pi:
                heading += 2 * math.pi

            self.heading = round(heading, 2)
            self.OdomUpdate = True


    def GetState(self):
        if self.InMission and self.laserUpdate and self.OdomUpdate:
            scan_range = []
            heading = self.heading
            min_range = 0.2
            collision = False
            finish = False

            for i in range(len(self.scan.ranges)):
                if self.scan.ranges[i] == float('Inf'):
                    scan_range.append(3.5)
                elif np.isnan(self.scan.ranges[i]):
                    scan_range.append(0)
                else:
                    scan_range.append(self.scan.ranges[i])

            obstacle_min_range = round(min(scan_range), 2)
            obstacle_angle = np.argmin(scan_range)
            if min_range > min(scan_range) > 0:
                collision = True

            current_distance = round(math.hypot(self.goal.pose.position.x - self.position.x, self.goal.pose.position.y - self.position.y), 2)
            if current_distance < 0.15:
                finish = True
            # return scan_range + [heading, current_distance], done, finish
            return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], collision, finish
        else:
            return None,None,None
    
    def MissionFinish(self):
        self.InMission = False

    

class TD3_Navigation(object):
    def __init__(self,load_directory,load_filename):
        rospy.init_node("td3_navigation")
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal',PoseStamped,self.SubGoal)
        self.scan_sub = rospy.Subscriber('scan',LaserScan,self.SubScan)
        self.odom_sub = rospy.Subscriber('odom',Odometry,self.SubOdom)
        self.speed_pub = rospy.Publisher("cmd_vel",Twist,queue_size=5)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        state_dim = 76
        action_dim = 2    #只控制旋转
        lr = 0.0
        max_action = 1.0
        self.td3_network = TD3(lr, state_dim, action_dim, max_action)
        self.td3_network.load(load_directory,load_filename)
        rate = rospy.Rate(20)
        self.stage = stage()
        self.isMoving = False
        self.listener=tf.TransformListener()
        self.listener.waitForTransform('map', 'odom', rospy.Time(0), rospy.Duration(5))
        rospy.loginfo("READY")
        while not rospy.is_shutdown():
            if self.stage.InMission:
                state,collision,finish = self.stage.GetState()
                if state == None:
                    continue
                if collision:
                    self.PubSpeed(0,0)
                    self.stage.MissionFinish()
                    self.reset()
                    rospy.loginfo("COLLISION ON WALLS")
                    
                elif finish:
                    self.PubSpeed(0,0)
                    self.stage.MissionFinish()
                    rospy.loginfo("FINISH TASK")
                else:
                    linear,angular = self.GetSpeed(np.asarray(state))
                    self.PubSpeed(linear,angular)
            else:
                pass
            rate.sleep()

    def reset(self,goal_x=0,goal_y=0):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

    def GetSpeed(self,state):
        time1 = time.time()
        action = self.td3_network.select_action(state)
        time2 = time.time()
        print("Use Time:{}".format(time2-time1))
        max_angle_vel = 2
        max_linear_spd = 0.3
        linear = action[1] * max_linear_spd/2 + max_linear_spd/2
        angular = action[0] * max_angle_vel
        return linear,angular

    def PubSpeed(self,linear,angular):
        speed = Twist()
        speed.linear.x = linear
        speed.angular.z = angular
        self.speed_pub.publish(speed)

    def SubGoal(self,pose):
        goal = self.listener.transformPose('odom',pose)
        if hasattr(self,"stage"):
            self.stage.UpdateGoal(goal)

    def SubScan(self,scan):
        if hasattr(self,"stage"):
            self.stage.UpdateScan(scan)

    def SubOdom(self, odom):
        if hasattr(self,"stage"):
            self.stage.UpdateOdometry(odom)
        
if __name__ == '__main__':
    load_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),"steer&spd/models117_laser72_2") # save trained models
    load_filename = "TD3_{}".format("stage2")
    td3 = TD3_Navigation(load_directory,load_filename)