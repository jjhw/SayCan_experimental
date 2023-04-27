#! /usr/bin/env python3

import numpy as np
import rospy
import actionlib

from geometry_msgs.msg import TransformStamped, Pose, Vector3, Quaternion
from sensor_msgs.msg import JointState
from wsg_50_common.srv import Move, MoveResponse
from std_srvs.srv import Trigger, TriggerResponse
from ur_dashboard_msgs.srv import IsProgramRunning, IsProgramRunningResponse
import tf2_ros
import tf_conversions
import cv2
from alpaca_move.msg import MovePointsAction, MovePointsGoal, MovePointsResult, MovePointsFeedback

from time import sleep
import threading
import moveit_commander


class Node:
    def __init__(self):
        rospy.init_node("robot_move")
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
        rospy.wait_for_service("/wsg_50_driver/move")
        self._gripper_srv(100, 420)
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
        self.server = actionlib.SimpleActionServer('~/move_by_camera', MovePointsAction, self.move_by_camera, False)
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
            print(self.pose_to_camera())
            rate.sleep()
            

    def move_by_camera(self, goal):
        prefix = "target_point"
        for i, point in enumerate(goal.poses):
            point_tf = TransformStamped()
            point_tf.header.frame_id = "camera"
            point_tf.header.stamp = rospy.Time.now()
            for i, item in enumerate(goal.poses):
                point_tf.child_frame_id = f"{prefix}_{i}"
                point_tf.transform.translation = Vector3(item.position.x, item.position.y, item.position.z)
                point_tf.transform.rotation = item.orientation
                self.tf_broadcaster.sendTransform(point_tf)
                tf_pos = self.tf_buffer.lookup_transform("base_link", f"{prefix}_{i}", rospy.Time(0), rospy.Duration(1.0))
                tf_ang = self.tf_buffer.lookup_transform("camera", f"{prefix}_{i}", rospy.Time(0), rospy.Duration(1.0))
                # create new tf for level out camera inclination
                tf = TransformStamped()
                tf.header.frame_id = "base_link"
                tf.header.stamp = rospy.Time.now()
                tf.child_frame_id = "tool"
                tf.transform.translation = tf_pos.transform.translation
                tf.transform.rotation = tf_ang.transform.rotation
                pose = Pose(position=tf.transform.translation, orientation=tf.transform.rotation)
                # publish goal
                goal_tf = TransformStamped()
                goal_tf.header.frame_id = "base_link"
                goal_tf.header.stamp = rospy.Time.now()
                goal_tf.child_frame_id = "goal"
                goal_tf.transform.translation = pose.position
                goal_tf.transform.rotation = pose.orientation
                self.tf_broadcaster.sendTransform(goal_tf)
                # set tcp to "tool0_center_link"
                self.move_group.set_pose_target(pose, "tool0")
                is_success = self.move_group.go(wait=True)
                if not is_success:
                    rospy.logerr("move failed")
                    self.server.set_aborted()
                    return
                self.server.publish_feedback(i/len(goal.poses))
        self.server.set_succeeded()

    def move_gripper(self, width):
        self._gripper_srv(width, 420)
    
    def pose_to_camera(self):
        pos_tf = self.tf_buffer.lookup_transform("camera", "tool0", rospy.Time(0), rospy.Duration(1.0))
        ang_tf = self.tf_buffer.lookup_transform("base_link", "tool0", rospy.Time(0), rospy.Duration(1.0))
        tf = TransformStamped()
        tf.header.frame_id = "tool"
        tf.header.stamp = rospy.Time.now()
        tf.child_frame_id = "camera"
        tf.transform.translation = pos_tf.transform.translation
        tf.transform.rotation = ang_tf.transform.rotation
        return tf

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
    tf_pub = CameraTFPublisher("/workspace/config/camera_pos.yaml")
    tf_pub.run()
    node.run()