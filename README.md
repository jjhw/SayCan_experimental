# Project on SayCan experiments: MISISRL&SRC
Proof of concept of the SayCan project applying on real UR5 robot

## Table of Contents
- [Overview](#overview)
- [Runnig the project](#running-the-project)


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
