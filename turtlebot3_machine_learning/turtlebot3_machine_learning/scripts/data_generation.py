#!/usr/bin/env python
# -*- coding:utf-8 -*-
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry,OccupancyGrid
from geometry_msgs.msg import PoseStamped,Twist
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from environment_icra import RespawnNew,MapInfo
from move_base_msgs.msg import MoveBaseAction,MoveBaseGoal,MoveBaseActionResult
import tf
import math
import os
import actionlib


  

class movementControl(object):
    def __init__(self,record):
        # rospy.Subscriber("/move_base/result",MoveBaseActionResult,self.resultCb)
        self.map_sub = rospy.Subscriber("/move_base/global_costmap/costmap",OccupancyGrid,self.mapCb)
        rospy.wait_for_message("/move_base/global_costmap/costmap",OccupancyGrid)
        self.respawn_goal = RespawnNew(self.map)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.respawn_goal.SetRange(0.5,8,0.5,4.4)
        self.client = actionlib.SimpleActionClient('/move_base',MoveBaseAction)
        self.dataRecorder = record


    def mapCb(self,map):
        if hasattr(self,"map"):
            pass
        else:
            self.map = MapInfo(map)

    def doneCb(self,status,b):
        if status == 3:
            self.PubGoal()
            self.dataRecorder.isRecord = True
        elif status == 4:
            rospy.wait_for_service('gazebo/reset_world')
            try:
                self.reset_proxy()
            except (rospy.ServiceException) as e:
                print("gazebo/reset_world service call failed")
            self.PubGoal()
            self.dataRecorder.isRecord = True

    def PubGoal(self):
        rospy.loginfo("PUB GOAL")
        goal_x,goal_y = self.respawn_goal.getPosition()
        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.stamp = rospy.Time.now()
        move_base_goal.target_pose.header.frame_id = "map"
        move_base_goal.target_pose.pose.position.x = goal_x
        move_base_goal.target_pose.pose.position.y = goal_y
        move_base_goal.target_pose.pose.orientation.z = 1
        self.client.send_goal(move_base_goal,done_cb=self.doneCb)
        self.dataRecorder.GoalCb(move_base_goal.target_pose)
    

class goal(object):
    def __init__(self):
        self.goal_x = None
        self.goal_y = None

    def setGoal(self,x,y):
        self.goal_x = x
        self.goal_y = y

    def state_cal(self,odom):
        assert(self.goal_x is not None)
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - position.y, self.goal_x - position.x)

        heading = goal_angle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        heading = round(heading, 2)
        current_distance = round(math.hypot(self.goal_x - position.x, self.goal_y - position.y), 3)
        return heading,current_distance


class DataRecorder(object):
    def __init__(self,rate=10):
        rospy.Subscriber("/scan",LaserScan,self.LaserCb)
        rospy.Subscriber("/odom",Odometry,self.OdomCb)
        rospy.Subscriber("/move_base_simple/goal",PoseStamped,self.GoalCb)
        rospy.Subscriber("/cmd_vel",Twist,self.CmdCb)
        self.isRecord = False
        self.control = movementControl(self)
        self.save_dir = os.path.join(os.getcwd(),"dataset.npy")
        self.tf_listener = tf.TransformListener()
        self.rate = rate
        self.RecordNum = 20000    #记录的数据量
        self.laserData = None
        self.odomData = None
        self.cmdData = None
        self.lastState = None
        self.goalData = goal()
        self.record_list = []
        
        self.CmdRecordConfig()
        r = rospy.Rate(rate)
        self.control.PubGoal()
        while not rospy.is_shutdown():
            self.recordData()
            if(len(self.record_list)>self.RecordNum):
                break
            r.sleep()
        np.save(self.save_dir,self.record_list)
        





    def LaserRecordConfig(self):
        self.laser_min = None
        self.laser_max = None
        self.laser_num = 60

        start_idx = -1
        end_idx = -1
        laser = self.laserData
        if self.laser_min is None:
            start_idx = 0
        else:
            start_idx = int((self.laser_min - laser.angle_min)/laser.angle_increment)
        if self.laser_max is None:
            end_idx = len(laser.ranges)
        else:
            end_idx = len(laser.ranges) - int((self.angle_max - laser.angle_max)/laser.angle_increment)
        assert(end_idx>start_idx)

        self.start_idx = start_idx
        self.end_idx = end_idx

    def CmdRecordConfig(self):
        self.max_angle_vel = 2
        self.max_linear_spd = 0.3
        
    
    def CmdCb(self,msg):
        self.cmdData = msg

    def LaserCb(self,msg):
        self.laserData = msg
        if not hasattr(self,'laser_num'):
            self.LaserRecordConfig()
        for data in self.laserData.ranges:
            if data == float('Inf'):
                data = 3.5
            elif np.isnan(data):
                data = 0
    
    def OdomCb(self,msg):
        self.odomData = msg

    def GoalCb(self,msg):
        goal_inodom = self.tf_listener.transformPose("odom",msg)
        self.goalData.setGoal(goal_inodom.pose.position.x,goal_inodom.pose.position.y)
        self.lastDis = round(math.hypot(self.goalData.goal_x - self.odomData.pose.pose.position.x, self.goalData.goal_y - self.odomData.pose.pose.position.y), 2)
        self.isRecord = True

    def laserRecord(self):
        state_laser = []
        delta = (self.end_idx-self.start_idx)/self.laser_num
        min_range = 10
        min_range_idx = -1
        for i in range(self.laser_num):
            idx = self.start_idx + int(delta*i)
            state_laser.append(self.laserData.ranges[idx])
            if(self.laserData.ranges[idx]<min_range):
                min_range_idx = i
                min_range = self.laserData.ranges[idx]
        return state_laser,min_range,min_range_idx

    def odomRecord(self):
        heading,current_dis = self.goalData.state_cal(self.odomData)
        return heading,current_dis

    def actionRecord(self):
        v_x = min(max(self.cmdData.linear.x,0),self.max_linear_spd)
        w = min(max(self.cmdData.angular.z,-self.max_angle_vel),self.max_angle_vel)
        action = [(v_x - self.max_linear_spd/2)/(self.max_linear_spd/2),w/self.max_angle_vel]
        return action


    def readyForRecord(self):
        if (self.laserData is not None) and (self.cmdData is not None) \
            and (self.odomData is not None) and (self.goalData.goal_x is not None):
            return True
        else:
            return False

    def recordData(self):
        if not self.readyForRecord():
            # rospy.loginfo("wait for msg")
            return 
        if self.isRecord:
            scan_range,obstacle_min_range,obstacle_angle = self.laserRecord()
            heading,current_dis = self.odomRecord()
            action = self.actionRecord()
            collision = False
            finish = False
            if obstacle_min_range < 0.2:
                collision = True
            if current_dis < 0.2:
                finish = True
            state = scan_range+[heading,current_dis,obstacle_min_range,obstacle_angle]
            reward = self.setReward(state,collision,finish,action)
            if self.lastState is None:
                self.lastState = state
            else:
                record = (self.lastState, action, reward, state, float(finish))
                self.record_list.append(record)
            if finish or collision:
                self.isRecord = False
                rospy.loginfo("stop")
                print("legth:",len(self.record_list))
                rospy.sleep(2)
                self.lastState = None
            


    def setReward(self, state, done,finish, action):
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]
        # angle = heading+action[0]*pi/8 +pi/2
        # tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * (angle) % (2 * math.pi) / math.pi)[0])   # tr->1 when angle->0 tr->-1 when angle->180
        move_dis = float((self.lastDis - current_distance ) * 100)
        self.lastDis = current_distance
        reward = 0

        reward = move_dis
        if obstacle_min_range < 0.25:
            reward = -30
        if done:
            rospy.loginfo("Collision!!")
            reward = -150

        if finish:
            rospy.loginfo("Goal!!")
            reward = 200
            

        return reward

            

if __name__ == "__main__":
    rospy.init_node("data_record")
    a = DataRecorder()