#! /usr/bin/env python3

import numpy as np
import rospy

from geometry_msgs.msg import TransformStamped, Pose, Vector3, Quaternion
from sensor_msgs.msg import JointState
from wsg_50_common.srv import Move, MoveResponse
from std_srvs.srv import Trigger, TriggerResponse
from ur_dashboard_msgs.srv import IsProgramRunning, IsProgramRunningResponse
import tf2_ros
import tf_conversions
from time import sleep
import moveit_commander


def move_robot(move_group, target=np.deg2rad([-140, -90, -140, 10, -330, 100])):
    # move robot to target joint position
    rospy.loginfo("robot current posX:\n" + str(move_group.get_current_pose()))
    rospy.loginfo(f"robot current posJ: {np.rad2deg(move_group.get_current_joint_values())} (deg)")
    rospy.loginfo(f"robot target posJ: {np.rad2deg(target)} (deg)")
    move_group.set_joint_value_target(target.tolist())
    is_success = move_group.go(wait=True)
    if not is_success:
        raise RuntimeError("move failed")

def close_gripper():
    _gripper_srv = rospy.ServiceProxy("/wsg_50_driver/move", Move)
    rospy.wait_for_service("/wsg_50_driver/move")
    rospy.loginfo("opening gripper...")
    _gripper_srv(15, 420)
    rospy.loginfo("closing gripper in 10 seconds")
    for i in range(10):
        rospy.loginfo(f"{10-i} seconds left")
        sleep(1)
    rospy.loginfo("closing gripper...")
    _gripper_srv(0, 420)

if __name__ == "__main__":
    rospy.init_node("robot_move")
    # wait for robot to be ready
    rospy.loginfo("waiting for robot to be ready...")
    # wait when ur script is running
    rospy.wait_for_service("/ur_hardware_interface/dashboard/program_running")
    rospy.wait_for_service("/ur_hardware_interface/dashboard/stop")
    rospy.wait_for_service("/ur_hardware_interface/dashboard/play")
    is_program_running = rospy.ServiceProxy("/ur_hardware_interface/dashboard/program_running", IsProgramRunning)
    stop = rospy.ServiceProxy("/ur_hardware_interface/dashboard/stop", Trigger)
    play = rospy.ServiceProxy("/ur_hardware_interface/dashboard/play", Trigger)
    stop()
    sleep(5)
    while True:
        rospy.loginfo("waiting for robot to be ready...")
        play()
        if not is_program_running().program_running:
            break
        sleep(1)
    move_group = moveit_commander.MoveGroupCommander("manipulator")
    move_robot(move_group)
    close_gripper()
    rospy.signal_shutdown("Done")
