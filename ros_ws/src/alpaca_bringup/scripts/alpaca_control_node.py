#! /usr/bin/env python3

import numpy as np
import rospy
import actionlib

from geometry_msgs.msg import TransformStamped, Pose, Vector3, Quaternion
from sensor_msgs.msg import JointState
from wsg_50_common.srv import Move, MoveResponse
from std_srvs.srv import Trigger, TriggerResponse, Empty, EmptyResponse
from ur_dashboard_msgs.srv import IsProgramRunning, IsProgramRunningResponse
import tf2_ros
import tf_conversions
import cv2
from alpaca_bringup.msg import MovePointsAction, MovePointsGoal, MovePointsResult, MovePointsFeedback
from typing import List, Tuple, Optional, Union, Callable
from time import sleep
import threading
import moveit_commander
import moveit_msgs.msg

gripper_speed = 100 # 420 is max

class Node:
    def __init__(self):
        rospy.init_node("robot_move")
        self._tcp_link = "wsg_50_center"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        # wait for robot to be ready
        rospy.loginfo("waiting for robot services to be ready...")
        # wait when ur script is running
        rospy.wait_for_service("/ur_hardware_interface/dashboard/program_running")
        rospy.wait_for_service("/ur_hardware_interface/dashboard/stop")
        rospy.wait_for_service("/ur_hardware_interface/dashboard/play")
        rospy.loginfo("robot services ready")
        rospy.loginfo("initializing gripper...")
        self._gripper_srv = rospy.ServiceProxy("/wsg_50_driver/move", Move)
        self._gripper_ackn_srv = rospy.ServiceProxy("/wsg_50_driver/ack", Empty)
        rospy.wait_for_service("/wsg_50_driver/move")
        self.gripper_open()
        rospy.loginfo("gripper initialized")
        rospy.loginfo("running robot program...")
        self.is_program_running = rospy.ServiceProxy("/ur_hardware_interface/dashboard/program_running", IsProgramRunning)
        self.stop = rospy.ServiceProxy("/ur_hardware_interface/dashboard/stop", Trigger)
        self.play = rospy.ServiceProxy("/ur_hardware_interface/dashboard/play", Trigger)
        self.stop()
        sleep(1)
        while not rospy.is_shutdown():
            rospy.loginfo("waiting for robot to be ready...")
            self.play()
            if self.is_program_running().program_running:
                break
            sleep(1)
        rospy.loginfo("robot ready")
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")

    def _init_services(self):
        # publish own services
        self.server = actionlib.SimpleActionServer('/alpaca/move_by_camera', MovePointsAction, self.move_by_camera, False)
        self._gripper_open_srv = rospy.Service("/alpaca/gripper/open", Trigger, self.gripper_open)
        self._gripper_close_srv = rospy.Service("/alpaca/gripper/close", Trigger, self.gripper_close)
        self._pose_to_camera_topic = rospy.Publisher("/alpaca/pose_to_camera", Pose, queue_size=1)
        self._display_trajectory_topic = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)
        self.server.start()

    def run(self):
        self._init_services()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.is_program_running().program_running:
                rospy.loginfo("robot stopped, restarting...")
                self.stop()
                sleep(5)
                self.play()
            # publish pose to camera
            self._pose_to_camera_topic.publish(self.pose_to_camera())
            rate.sleep()

    def move_by_camera(self, goal):
        time = rospy.Time.now()
        result = MovePointsFeedback()
        result.progress.data = 0
        self.server.publish_feedback(result)
        path = self._targets_to_path(goal.poses, time)
        (plan, fraction) = self.move_group.compute_cartesian_path(path, 0.01, 0.0)
        is_success = True if fraction > 0.95 else False
        if not is_success:
            rospy.logerr(f"cartesian path planning failed, fraction: {fraction}")
            result = MovePointsResult()
            result.success.data = False
            self.server.set_succeeded(result)
            return
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.move_group.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self._display_trajectory_topic.publish(display_trajectory);
        ret = self.move_group.execute(plan, wait=True)
        result = MovePointsResult()
        result.success.data = ret
        self.server.set_succeeded(result)
    
    def _targets_to_path(self, targets: List[Pose], time: rospy.Time) -> List[Pose]:
        assert len(targets) > 0, "no targets given"
        assert isinstance(targets, list), "targets must be a list"
        assert all([isinstance(target, Pose) for target in targets]), "targets must be a list of Pose"
        prefix_goal_camera = "target_point"
        prefix_goal_base = "target_point_base"
        path = [self.move_group.get_current_pose(self._tcp_link).pose]
        for i, item in enumerate(targets):
            point_tf = TransformStamped()
            point_tf.header.frame_id = "camera"
            point_tf.header.stamp = time
            point_tf.child_frame_id = f"{prefix_goal_camera}_{i}"
            point_tf.transform.translation = Vector3(item.position.x, item.position.y, item.position.z)
            point_tf.transform.rotation = item.orientation
            self.tf_broadcaster.sendTransform(point_tf)
            tf_pos = self.tf_buffer.lookup_transform("base_link", f"{prefix_goal_camera}_{i}", time, rospy.Duration(1.0))
            tf_ang = self.tf_buffer.lookup_transform("camera", f"{prefix_goal_camera}_{i}", time, rospy.Duration(1.0))
            # create new tf for level out camera inclination
            tf = TransformStamped()
            tf.header.frame_id = "base_link"
            tf.header.stamp = time
            tf.child_frame_id = "tool"
            tf.transform.translation = tf_pos.transform.translation
            tf.transform.rotation = tf_ang.transform.rotation
            pose = Pose(position=tf.transform.translation, orientation=tf.transform.rotation)            
            path.append(pose)
        for i, point in enumerate(path):
            goal_tf = TransformStamped()
            goal_tf.header.frame_id = "base_link"
            goal_tf.header.stamp = time
            goal_tf.child_frame_id = f"{prefix_goal_base}_{i}"
            goal_tf.transform.translation = point.position
            goal_tf.transform.rotation = point.orientation
            self.tf_broadcaster.sendTransform(goal_tf)
        return path
    
    def gripper_open(self, req=None):
        self._gripper_ackn_srv()
        self._gripper_srv(108, gripper_speed)
        return TriggerResponse(True, "gripper opened")
    
    def gripper_close(self, req=None):
        self._gripper_ackn_srv()
        self._gripper_srv(20, gripper_speed)
        return TriggerResponse(True, "gripper closed")
    
    def pose_to_camera(self) -> Pose:
        pos_tf = self.tf_buffer.lookup_transform("camera", self._tcp_link, rospy.Time(0), rospy.Duration(1.0))
        rot_tf = self.tf_buffer.lookup_transform("base_link", self._tcp_link, rospy.Time(0), rospy.Duration(1.0))
        tf = TransformStamped()
        tf.header.frame_id = "camera"
        tf.header.stamp = rospy.Time.now()
        tf.child_frame_id = "tool_to_camera"
        tf.transform.translation = pos_tf.transform.translation
        tf.transform.rotation = rot_tf.transform.rotation
        self.tf_broadcaster.sendTransform(tf)

        pose = Pose()
        pose.position = pos_tf.transform.translation
        pose.orientation = rot_tf.transform.rotation
        return pose


class CameraTFPublisher:
    def __init__(self, calbration_file):
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.calibration_file = calbration_file
        self.camera_pos, self.camera_ang = self.load_params(calbration_file)
        self.cam_br = tf2_ros.StaticTransformBroadcaster()
        self.rate = rospy.Rate(10)
        self.camera_frame = "camera"
        self.base_frame = "base"
        self.camera_to_base = self.get_camera_to_base()
        self._thread = None

    def get_camera_to_base(self):
        camera_tf = TransformStamped()
        camera_tf.header.stamp = rospy.Time.now()
        camera_tf.header.frame_id = self.base_frame
        camera_tf.child_frame_id = self.camera_frame
        camera_tf.transform.translation = Vector3(*self.camera_pos)
        camera_tf.transform.rotation = Quaternion(*self.camera_ang)
        return camera_tf
    
    def run(self):
        self._thread = threading.Thread(target=self.loop)
        self._thread.start()

    def loop(self):
        while not rospy.is_shutdown():
            self.cam_br.sendTransform(self.camera_to_base)
            self.rate.sleep()

    def load_params(self, filename):
        cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        camera_pos = cv_file.getNode("camera_pos").mat()
        camera_ang = cv_file.getNode("camera_ang").mat()
        if camera_pos is None or camera_ang is None :
            rospy.logwarn('Calibration file "' + filename + '" doesn\'t exist or corrupted')
            return np.zeros((3), dtype=np.float64), np.zeros((4), dtype=np.float64)
        return camera_pos, camera_ang

if __name__ == "__main__":
    node = Node()
    calibration_file_path = rospy.get_param("~calib_path")
    tf_pub = CameraTFPublisher("/workspace/config/camera_pos.yaml")
    tf_pub.run()
    node.run()