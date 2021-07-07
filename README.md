# robot_elevator_strategy_planning

## Environmental Setup for ROS Melodic
### Turtle Bot 3 Installation 
1. Install turtle bot 3 packages
```
cd src
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
cd ..
catkin_make
```
2. Install turtle bot simulation package
```
cd src
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ..
catkin_make
```
3. Choose turtle bot model
```
# Available model: burger / waffle / waffle_pi
export TURTLEBOT3_MODEL = burger 
```

## Starting the gazebo world (basic)
1. Empty world which only contain turtlebot
```
export TURTLEBOT3_MODEL = burger 
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```
2. Control turtlebot with keyboard
```
export TURTLEBOT3_MODEL = burger 
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
When you can control the turtlebot with keyboard, it means the basic installtion is correct.