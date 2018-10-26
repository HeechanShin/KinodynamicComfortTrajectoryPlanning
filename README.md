# KinodynamicComfortTrajectoryPlanning
Implementation of a paper "Kinodynamic Comfort Trajectory Planning for Car-like Robots" (IROS18)
https://sglab.kaist.ac.kr/KinodynamicComfortPlanning/

# Requirements
* Ipopt ( https://projects.coin-or.org/Ipopt )
* V-Rep ( http://www.coppeliarobotics.com/ )
* ROS ( http://www.ros.org/ )
* V-Rep - ROS Interface ( https://github.com/CoppeliaRobotics/v_repExtRosInterface )
* EIGEN ( https://eigen.tuxfamily.org/dox/ )
* Ackermann_msgs ( http://wiki.ros.org/ackermann_msgs )

# Tested Environment
* Ubuntu 16.04
* Ipopt 3.12.11
* V-Rep 3.5
* ROS Kinetic

* EIGEN 3

# Getting Started
1. Install requirements

    1. Installing the Ipopt by source code is recommended.
    2. Over version 3.5 of V-Rep is recommended.
    3. ROS, EIGEN, Ackermann_msgs can be downloaded by apt-get in Ubuntu.
    4. Build RosInterface and move built library to V-Rep root folder.
  
2. Install This repository
```
  cd ~/catkin_ws/src ( your catkin workspace source file folder )
  git clone https://github.com/HeechanShin/KinodynamicComfortTrajectoryPlanning.git
  cd ../
  catkin_make
  source devel/setup.bash
  ```
  
3. Launch V-Rep
```
  cd ~/vrep ( your V-Rep root )
  ./vrep.sh
```
open included test scene, which is in the scene folder and start simulation.

4. Launch comfort_trajectory_optimizer_node
```
  rosrun comfort_trajectory_optimizer comfort_trajectory_optimizer_node
  ```
