#! /usr/bin/env python
import rospy
import rospkg 
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import random
import math
import numpy as np

tick_sign = u'\u2714'.encode('utf8')
cross_sign = u'\u274c'.encode('utf8')

# World Configuration
# RED -> X GREEN -> Y BULE -> Z
obstacles = ['cylinder','cylinder_clone','cylinder_clone_0','cylinder_clone_1']
turtlebot = 'turtlebot3_waffle'
elevator = 'cave'
# (x,y,z,r,p,y)
turtlebot_init_pos = (0,1.0,0,0,0,3.14)
elevator_init_pos = (-1.99,1.048,0,0,0,0)
cave_y_max = 1.55
cave_y_min = 0.4
cave_x_max = -1.4
cave_x_min = -2.4
cave_z = 0.164
objects = []

# Euler Quaternian Conversion
def euler_to_quaternion(r):
    (roll,pitch,yaw) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return qx, qy, qz, qw

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
    x = random.uniform(cave_x_min,cave_x_max)
    y = random.uniform(cave_y_min,cave_y_max)
    z = cave_z
    return x,y,z

def main():
    # Reset all the status in the world of gazebo
    rospy.wait_for_service('/gazebo/reset_world')
    reset_world = rospy.ServiceProxy('/gazebo/reset_world',Empty)
    reset_world()
    print("The world is reset ... " + tick_sign)

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

if __name__ == '__main__':
    rospy.init_node('reset_world')
    main()

