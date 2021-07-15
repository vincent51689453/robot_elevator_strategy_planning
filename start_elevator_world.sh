#!/bin/bash
sudo killall rosmaster
sudo killall gzserver
sudo killall gzclient
source devel/setup.bash
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_gazebo turtlebot3_elevator_world.launch