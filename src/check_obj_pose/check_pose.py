#! /usr/bin/env python
from __future__ import division
import rospy
import rospkg 
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import *
import random
import math
import numpy as np
from datetime import datetime

object_name = 'cave'

get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
model = GetModelStateRequest()
model.model_name = object_name
objstate = get_state_service(model)
x = objstate.pose.position.x
y = objstate.pose.position.y
z = objstate.pose.position.z
print("target:{} x:{} y:{} z:{}".format(object_name,x,y,z))