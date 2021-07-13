#!/usr/bin/env python
"""
Robot Control
0 - Forward
1 - Left
2 - Right
3 - Stop
4 - Backward
"""
#System Packages
from __future__ import division
import sys
import os
import cv2
import numpy as np
import numpy.ma as ma
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
depth_display_image = None
tick_sign = u'\u2714'.encode('utf8')
cross_sign = u'\u274c'.encode('utf8')
# Safety depth marker
marker1 = (int(1920/2), int(1080/2))
marker2 = (int(1920/2)-100, int(1080/2))
marker3 = (int(1920/2)+100, int(1080/2))
markers_z = []

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
eps_decay = 0.996
action_list = ['Forward','Left','Right','Stop','Backward']

def depth_callback(ros_msg):
    # Depth image callback
    global depth_display_image,marker_z

    bridge = CvBridge()
    depth_image = None
    # Use cv_bridge() to convert the ROS image to OpenCV format
    try:
        # The depth image is a single-channel float32 image
        depth_image = bridge.imgmsg_to_cv2(ros_msg, "passthrough")
    except CvBridgeError, e:
        print e
    # Convert the depth image to a Numpy array since most cv2 functions
    # require Numpy arrays.
    depth_array = np.array(depth_image, dtype=np.float32)
    d1 = depth_image[marker1[1],marker1[0]]
    d2 = depth_image[marker2[1],marker2[0]]
    d3 = depth_image[marker3[1],marker3[0]]
    markers_z.append(d1)
    markers_z.append(d2)
    markers_z.append(d3)
    # Normalize the depth image to fall between 0 (black) and 1 (white)
    cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
    # Process the depth image
    depth_display_image = nan_recover(depth_array)

# Replace all NAN by 0
def nan_recover(frame):
    # NAN removeal
    # Search NAN by height (y)
    for i in range(0,frame.shape[0]):
        # Search NAN by width (x)
        for j in range(0,frame.shape[1]):
            # Matrix(y,x)
            if np.isnan(frame[i,j]):
                frame[i,j] = 0
    return frame

def main():
    global depth_display_image,scores,markers_z

    # Subscribe depth image
    rospy.Subscriber(depth_image_topic,Image,callback=depth_callback, queue_size=1)
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
            print("")
            print("#### Epoch: " + str(i)+ " ####")
            while (t<max_dt):
                # Current state
                if(RL_mode==0):
                    #print("Mode: Current state -> Choosing Action")
                    # Set state as depth image observation (current state)
                    #depth_display_image = cv2.resize(depth_display_image,(720,480))
                    image_tensor = transformer(depth_display_image)
                    state = Variable(image_tensor).cuda()  
                    state = torch.unsqueeze(image_tensor,0)

                    # Select an action
                    action = robot.act(state,eps)
                    #print("Agent chosen action: " + action_list[action])
                    RL_mode = 1
                # Next state
                else:
                    # Set state as depth image observation (next state)
                    #depth_display_image = cv2.resize(depth_display_image,(720,480))
                    image_tensor = transformer(depth_display_image)
                    next_state = Variable(image_tensor).cuda()  
                    next_state = torch.unsqueeze(image_tensor,0)

                    #print("Mode: Next state -> Performing Action")
                    # Apply to the environment
                    reward = environment.perform(action,0.5,0.5,dt=1000,mark_depth=markers_z)
                    #print("Reward at t->{}= {}".format(str(t),str(reward)))
                    print("Epoch:{} Batch [{}/{}]: Action->{} Reward->{}".format(str(i),str(t),str(max_dt),action_list[action],str(reward)))
                    

                    # Save experience
                    robot.step(state,action,reward,next_state,True)
                    RL_mode = 0
                    t += 1
                    # Decrease epsilon
                    eps = max(eps*eps_decay,eps_end)
            environment.reset_env()
            time.sleep(1)
            t = 0
            i += 1
           
            # Visualization in new window
            #depth_display_image = cv2.resize(depth_display_image,(720,480))
            #cv2.imshow("Depth image",depth_display_image)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

if __name__ == '__main__':
    rospy.init_node(node_name)
    print(node_name+" is started!")
    main()
