#!/usr/bin/env python
"""
Robot Control
0 - Forward
1 - Left
2 - Right
3 - Backward
4 - Stop
"""
#System Packages
import sys
import os
import cv2
import numpy as np
import time

#ROS Packages
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

#Pytorch Packages
import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable

#Customized Packages
import environment
import RLagent
import DQN

# ROS Topic
node_name = 'RL_Controller'
depth_image_topic = '/camera/depth/image_raw'

# Variables
depth_image = None
depth_display_image = None
bridge = CvBridge()
tick_sign = u'\u2714'.encode('utf8')
cross_sign = u'\u274c'.encode('utf8')
marker = (1920/2, 1080/2)
marker_z = 0

"""
DQN Hyperparameters
max_epoch (int): maximum number of training epsiodes
max_dt (int): maximum number of timesteps per episode
move_dt(int): maximum number of timesteps per action
eps_start (float): starting value of epsilon, for epsilon-greedy action selection
eps_end (float): minimum value of epsilon 
eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
scores (integer): list containing score from each epoch
"""
max_epoch = 200
max_dt = 20
move_dt = 2000
eps_start = 1.0
eps_end = 0.01
eps_decay=0.996
action_list = ['Forward','Left','Right','Backward','Stop']

def depth_callback(ros_msg):
    # Depth image callback
    global depth_image,bridge,depth_display_image,marker_z
    # Use cv_bridge() to convert the ROS image to OpenCV format
    try:
        # The depth image is a single-channel float32 image
        depth_image = bridge.imgmsg_to_cv2(ros_msg, "32FC1")
        print(depth_image.shape)
    except CvBridgeError, e:
        print e
    # Convert the depth image to a Numpy array since most cv2 functions
    # require Numpy arrays.
    depth_array = np.array(depth_image, dtype=np.float32)
    marker_z = depth_image[marker[1],marker[0]]
    # Normalize the depth image to fall between 0 (black) and 1 (white)
    cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
    # Process the depth image
    depth_display_image = process_depth_image(depth_array)

def process_depth_image(frame):
    # Just return the raw image for this demo
    return frame

def main():
    global depth_display_image,bridge,scores

    # Subscribe depth image
    rospy.Subscriber(depth_image_topic,Image,callback=depth_callback)
    print("Depth image subscriber ... " + tick_sign)

    # Setup RL agent
    # Action: Forward/Left/Right/Backward/Stop
    # State: Depth image
    robot = RLagent.Agent(state_size=1,action_size=5,seed=0)
    print("Reinforcement Agent setup ... " + tick_sign)


    # Training initilization
    # Reset the world
    environment.reset_env()
    time.sleep(1)
    reward = 0
    scores = []
    eps = eps_start
    transformer = transforms.ToTensor()
    # 0 for current state and 1 for next state
    RL_mode = 0

    # Start training
    i = 1
    t = 0
    total_reward = 0
    while(i<max_epoch):
        if(depth_display_image is not None):
            global marker_z
            print("Depth:",marker_z)
           
            # Visualization in new window
            #depth_display_image = cv2.resize(depth_display_image,(720,480))
            #cv2.imshow("Depth image",depth_display_image)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

if __name__ == '__main__':
    rospy.init_node(node_name)
    print(node_name+" is started!")
    main()
