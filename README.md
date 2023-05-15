# Project on SayCan experiments: MISISRL&SRC
Proof of concept of the SayCan project applying on real UR5 robot

## Table of Contents
- [Overview](#overview)
- [Runnig the project](#running-the-project)
- [Camera calibration](#camera-calibration)
- [Bringing up](#bringing-up)

## Overview
We are using *ROS Noetic* to set up communication between services. UR5 controller is provided by *MoveIt!* for safe robot motion. The stand is equipped with a *Realsense L515* camera for object segmentation.
## Running the project
This project uses git submodules. To clone the project with all submodules, use the following command:
```bash
git clone https://github.com/atokagzx/SayCan_experimental -b master --recurse-submodules
```
We recommend to use a docker container to run the project. To do so, you need to install [**docker**](https://docs.docker.com/engine/install/ubuntu/) on your machine.
 Then, you can run the project with the following commands in terminal:
```bash
docker/build_docker.sh
docker/run_docker.sh
```
you are able to open a new terminal session inside docker container with the following command:
```bash
docker/into_docker.sh
```
roslaunch ur_calibration calibration_correction.launch robot_ip:=<robot ip> target_filename:=/workspace/ros_ws/src/alpaca_moveit_config/stand_model/kinematics.yaml

## Camera calibration
For the correct work of the project, you need to calibrate the camera extrinsics.  
- First of all you should print the [calibration pattern](/docs/arucoboard.pdf) on a sheet of paper.  
- Then you need to run the following command in terminal:
```bash
roslaunch tf_broadcaster calibrate_camera_position.launch
```
This command wil bring up the camera node, robot node and tf broadcaster node. Robot will move to the calibration position and the camera will start to capture images. Then you will have 10 secods to insert the calibration pattern into the gripper. When the calibration pattern is in the gripper, you need to press the ***S*** in the OpenCV window. After that, the calibration node will save the calibration file.
Default path to the calibration file is:`/workspace/config/camera_pos.yaml`

## Bringing up
To bringup the project, you need to run the following commands in terminal:
```bash
roslaunch alpaca_bringup bringup.launch
```
This command will start all the necessary nodes for the project. Including:
- **UR5 controller** - *MoveIt!* node for robot motion
- **Alpaca controller** - *Alpaca control* node provides the main logic of the stand control
- **Realsense camera** - *Realsense* node for camera data
- **WSG50 gripper** - *WSG50* node for gripper control

Next you are able to run example scripts to test the project:
```bash
rosrun alpaca_connector monitoring.py # monitoring of robot's ee position to camera frame
rosrun alpaca_connector control_gripper.py # control gripper example script
rosrun alpaca_connector follow_trajectory.py # execute trajectory using robot's ee position to camera frame
```
