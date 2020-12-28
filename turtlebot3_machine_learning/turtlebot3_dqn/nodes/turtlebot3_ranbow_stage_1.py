#!/usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import sys
from ranbow_dqn import DQNAgent
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment_stage_1 import Env

EPISODES = 3000


