# robot_elevator_strategy_planning

## Environmental Setup for ROS Melodic
### ROS multiplot Installation
```
sudo apt-get update
sudo apt-get install ros-melodic-rqt-multiplot
```

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
export TURTLEBOT3_MODEL=burger 
```

## Starting the gazebo world (basic)
1. Empty world which only contain turtlebot
```
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```
2. Control turtlebot with keyboard
```
export TURTLEBOT3_MODEL=waffle 
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
When you can control the turtlebot with keyboard, it means the basic installtion is correct.

## Reinforcemnet Learning Simulation
Deep Q Network approach is chosen as the control algorithm by defining the states, rewards and actions in Gazebo. The state is the depth image captured by the depth camera such as realsense, Zed camera, etc. The size of the action vector equals to 5 including forward, backward, left, right and stop. Each action associate with the state can generate an unique Q values. As the result, we are trying to plan the strategy based on maximizing the Q values. Furthremore, in order to guide the robot and define the achievemnt of it, a reward function is defined based on the distance between the robot and the elevator and the distance between several objects.

![image](https://github.com/vincent51689453/robot_elevator_strategy_planning/blob/rtx-melodic/git_image/DQN_Diagram.JPG)

1. Launch elevator world
```
export TURTLEBOT3_MODEL=waffle 
./start_elevator_world.sh
```

2. Reset elevator world and randomly spawn robot and obstacles
```
export TURTLEBOT3_MODEL=waffle 
./reset_elevator_world.sh
```

3. Training the robot with reinforcement learning
```
./train_elevator.sh
```

![image](https://github.com/vincent51689453/robot_elevator_strategy_planning/blob/rtx-melodic/git_image/elevator_world_3.png)

![image](https://github.com/vincent51689453/robot_elevator_strategy_planning/blob/rtx-melodic/git_image/basic_RL_demo.gif)

