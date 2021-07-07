#! /usr/bin/env python
import rospy
import rospkg 
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import random

tick_sign = u'\u2714'.encode('utf8')
cross_sign = u'\u274c'.encode('utf8')

# World Configuration
# RED -> X GREEN -> Y BULE -> Z
obstacles = ['cylinder','cylinder_clone','cylinder_clone_0','cylinder_clone_1']
turtlebot = 'turtlebot3_waffle'
turtlebot_init_pos = (0,1.0,0,0,0,0,0)
cave_y_max = 1.55
cave_y_min = 0.4
cave_x_max = -1.4
cave_x_min = -2.4
cave_z = 0.164

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
    state_msg.pose.orientation.x = turtlebot_init_pos[3]
    state_msg.pose.orientation.y = turtlebot_init_pos[4]
    state_msg.pose.orientation.z = turtlebot_init_pos[5]
    state_msg.pose.orientation.w = turtlebot_init_pos[6]
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)
        print("Render " + turtlebot + " done ... " + tick_sign)
    except rospy.ServiceException, e:
        print("Render " + turtlebot + " failed ... " + cross_sign)

if __name__ == '__main__':
    rospy.init_node('reset_world')
    main()

