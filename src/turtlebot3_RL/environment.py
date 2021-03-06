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

tick_sign = u'\u2714'.encode('utf8')
cross_sign = u'\u274c'.encode('utf8')

# World Configuration
# RED -> X GREEN -> Y BULE -> Z
obstacles = ['cylinder','cylinder_clone','cylinder_clone_0','cylinder_clone_1']
turtlebot = 'turtlebot3_waffle'
elevator = 'cave'
# (x,y,z,r,p,y)
turtlebot_init_pos = (0,1.0,0,0,0,3.14)
elevator_init_pos = (-1.99025,1.04874,0,0,-0,0)
cave_y_max = 1.4
cave_y_min = 0.6
cave_x_max = -1.4
cave_x_min = -2.3
cave_z = 0.164
objects = []

dt_counter = 0
r = 0

# Get turtle bot pose
def where_is_it(object_name):
    get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    model = GetModelStateRequest()
    model.model_name = object_name
    objstate = get_state_service(model)
    x = objstate.pose.position.x
    y = objstate.pose.position.y
    z = objstate.pose.position.z
    return x,y,z

# Euler Quaternian Conversion
def euler_to_quaternion(r):
    (roll,pitch,yaw) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return qx, qy, qz, qw

# Euler Quaternian Conversion
def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw, pitch, roll

# Generate random position for an obstacle
def shuffle_pos():
    x, y, z = 0, 0, 0
    global cave_x_max,cave_x_min
    global cave_y_max,cave_y_min
    mils = int(str(datetime.now())[20:])
    random.seed(mils)
    x = random.uniform(cave_x_min,cave_x_max)
    y = random.uniform(cave_y_min,cave_y_max)
    z = cave_z
    return x,y,z

# Reset everything (deprecated)
def reset_env():
    print("\r\n")
    # Reset all the status in the world of gazebo
    rospy.wait_for_service('/gazebo/reset_world')
    reset_world = rospy.ServiceProxy('/gazebo/reset_world',Empty)
    reset_world()
    print("The world is reset ... " + tick_sign)
    global objects
    objects = []
    seed_i = 0
    first_x,first_y = 0,0
    
    # Rearrnage obstacles in the cave
    for i in range(0,len(obstacles)):
        # Generate random locations
        x,y,z = 0,0,0
        x,y,z = shuffle_pos()

        # Publish new object location and orientation
        state_msg = ModelState()
        state_msg.model_name = obstacles[i]
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0
        rospy.wait_for_service('/gazebo/set_model_state')
        objects.append((x,y,z))
        #print(state_msg)
        #print("object:{} x:{} y:{} z:{}".format(obstacles[i],x,y,z))
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            print("Render " + obstacles[i] + " done ... " + tick_sign)
        except rospy.ServiceException, e:
            print("Render " + obstacles[i] + " failed ... " + cross_sign)
    
    # Set turtle bot position
    state_msg = ModelState()
    state_msg.model_name = turtlebot
    state_msg.pose.position.x = turtlebot_init_pos[0]
    state_msg.pose.position.y = turtlebot_init_pos[1]
    state_msg.pose.position.z = turtlebot_init_pos[2]
    euler = (turtlebot_init_pos[3],turtlebot_init_pos[4],turtlebot_init_pos[5])
    x,y,z,w = euler_to_quaternion(euler)
    state_msg.pose.orientation.x = x
    state_msg.pose.orientation.y = y
    state_msg.pose.orientation.z = z
    state_msg.pose.orientation.w = w
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)
        print("Render " + turtlebot + " done ... " + tick_sign)
    except rospy.ServiceException, e:
        print("Render " + turtlebot + " failed ... " + cross_sign)

    # Set cave position
    state_msg = ModelState()
    state_msg.model_name = elevator
    state_msg.pose.position.x = elevator_init_pos[0]
    state_msg.pose.position.y = elevator_init_pos[1]
    state_msg.pose.position.z = elevator_init_pos[2]
    euler = (elevator_init_pos[3],elevator_init_pos[4],elevator_init_pos[5])
    x,y,z,w = euler_to_quaternion(euler)
    state_msg.pose.orientation.x = x
    state_msg.pose.orientation.y = y
    state_msg.pose.orientation.z = z
    state_msg.pose.orientation.w = w
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)
        print("Render " + elevator + " done ... " + tick_sign)
    except rospy.ServiceException, e:
        print("Render " + elevator + " failed ... " + cross_sign)  


# Apply chosen action to the gazebo world
def perform(action='turtlebot3_waffle',basic_power=0.5,turn_power=0.5,max_dt=3):
    global objects
    global dt_counter
    global r
    task_complete = False

    # Get turtle bot position
    robot_x,robot_y,robot_z = where_is_it(turtlebot) 
    # Get obstacles position
    for i in range(0,4):
        a,b,c = where_is_it(obstacles[i])
        objects.append((a,b,cave_x_max))

    # ROS Publisher (/cmd_vel)
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    speed_msg = Twist()    
    if (action == 0):
        speed_msg.linear.x = basic_power*1
        speed_msg.linear.y = 0
        speed_msg.linear.z = 0
        speed_msg.angular.x = 0
        speed_msg.angular.y = 0
        speed_msg.angular.z = 0
    elif (action == 1):
        speed_msg.linear.x = basic_power
        speed_msg.linear.y = 0
        speed_msg.linear.z = 0
        speed_msg.angular.x = 0
        speed_msg.angular.y = 0
        speed_msg.angular.z = turn_power
    elif (action == 2):
        speed_msg.linear.x = basic_power
        speed_msg.linear.y = 0
        speed_msg.linear.z = 0
        speed_msg.angular.x = 0
        speed_msg.angular.y = 0
        speed_msg.angular.z = -turn_power   
    elif (action == 3):     
        speed_msg.linear.x = 0
        speed_msg.linear.y = 0
        speed_msg.linear.z = 0
        speed_msg.angular.x = 0
        speed_msg.angular.y = 0
        speed_msg.angular.z = 0    
    else:
        speed_msg.linear.x = -basic_power*1
        speed_msg.linear.y = 0
        speed_msg.linear.z = 0
        speed_msg.angular.x = 0
        speed_msg.angular.y = 0
        speed_msg.angular.z = 0    
    # Publish the cmd_vel message
    velocity_publisher.publish(speed_msg)     

    #Reward function
    # Calculate euclidean distances
    cave_mid_x = cave_x_max+cave_x_min/2
    cave_mid_y = cave_y_max+cave_y_min/2
    d_cave = ((cave_mid_x-robot_x)**2+(cave_mid_y-robot_y)**2)**0.5
    d_obj1 = ((cave_mid_x-objects[0][0])**2+(cave_mid_y-objects[0][1])**2)**0.5
    d_obj2 = ((cave_mid_x-objects[1][0])**2+(cave_mid_y-objects[1][1])**2)**0.5
    d_obj3 = ((cave_mid_x-objects[2][0])**2+(cave_mid_y-objects[2][1])**2)**0.5
    d_obj4 = ((cave_mid_x-objects[3][0])**2+(cave_mid_y-objects[3][1])**2)**0.5

    # Calculate relative orientation
    # Set the robot as the center (0,0)
    if(cave_mid_y==robot_y):
        theta = 0
    else:
        theta = math.atan(abs(cave_mid_y-robot_y)/abs(cave_mid_x-robot_x))
        theta = int(math.degrees(theta))

    # Reward 
    basic_r = 1/d_cave*10
    penalty_distance = 1/(d_obj1+d_obj2+d_obj3+d_obj4)*5
    penalty_deflection = theta*10
    extreme = 0
    
    # Extreme reward/penalty
    if(robot_x<cave_x_max)and(robot_x>cave_x_min):
        if(robot_y<cave_y_max)and(robot_y>cave_y_min):
            if(action==3):
                # Extreme reward if the robot can stop inside the cave
                extreme = 9000
                task_complete = True

    if((robot_x>cave_x_max)or(robot_x<cave_x_min))or((robot_y>cave_y_max)or(robot_y<cave_y_min)):
        if(action==3)or(action==4):
            # Extreme penalty if the robot stops/goes backward outside the cave
            extreme = -200
    
    if(dt_counter<=max_dt):
        r += basic_r - penalty_deflection - penalty_distance + extreme
        dt_counter += 1
    else:
        dt_counter = 0
        r = basic_r - penalty_deflection - penalty_distance + extreme

    # Avoid negative reward
    if r < 0:
        r = (r*-1)/2
    # Reward Analysis
    """
    print("Reward Analysis:")
    weight_distance = str(round((penalty_distance/basic_r),2)*100*-1)+'%'
    weight_deflection = str(round((penalty_deflection/basic_r),2)*100*-1)+'%'
    print("Reward:{} Distance Penalty:{} Deflection Penalty:{}".format(basic_r,penalty_distance,penalty_deflection))
    print("Distance Penalty Discount:{} Deflection Penalty Discount:{}\r\n".format(weight_distance,weight_deflection))
    """

    # Info for ros plot
    info_packet = (d_cave,d_obj1,d_obj2,d_obj3,d_obj4,theta)

    return r,task_complete,info_packet

# Get obstacles vector from gazebo
def observe_gazebo():
    # Object coordinates (x,y,z)
    object_coor = []    
    d_cave = 0
    d_obj1,d_obj2,d_obj3,d_obj4 = 0,0,0,0
    theta = 0

    # Get obstacles position
    for i in range(0,4):
        a,b,c = where_is_it(obstacles[i])
        object_coor.append((a,b,cave_z))

    # Get turtle bot position
    robot_x,robot_y,robot_z = where_is_it(turtlebot)

    # Calculate euclidean distances
    cave_mid_x = cave_x_max+cave_x_min/2
    cave_mid_y = cave_y_max+cave_y_min/2
    d_cave = ((cave_mid_x-robot_x)**2+(cave_mid_y-robot_y)**2)**0.5
    d_obj1 = ((cave_mid_x-object_coor[0][0])**2+(cave_mid_y-object_coor[0][1])**2)**0.5
    d_obj2 = ((cave_mid_x-object_coor[1][0])**2+(cave_mid_y-object_coor[1][1])**2)**0.5
    d_obj3 = ((cave_mid_x-object_coor[2][0])**2+(cave_mid_y-object_coor[2][1])**2)**0.5
    d_obj4 = ((cave_mid_x-object_coor[3][0])**2+(cave_mid_y-object_coor[3][1])**2)**0.5

    # Calculate relative orientation
    # Set the robot as the center (0,0)
    if(cave_mid_y==robot_y):
        theta = 0
    else:
        theta = math.atan(abs(cave_mid_y-robot_y)/abs(cave_mid_x-robot_x))
        theta = int(math.degrees(theta))

    output_vector = [d_cave*10,d_obj1*10,d_obj2*10,d_obj3*10,d_obj4*10,theta]
    return output_vector