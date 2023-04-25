#!/usr/bin/env python3
from copy import deepcopy
from datetime import datetime

from numpy import pi
import rospy
import numpy as np
import cv2
from cv2 import aruco
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, TransformStamped, Vector3, Quaternion

import tf2_ros
import tf_conversions
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3

static_markers_sked = list(range(12))# list of static marker ids
import numpy as np
import pyrealsense2
import numpy as np
import cv2
from typing import List, Tuple

def to_real(points, depth_map, cameraInfo):
    if cameraInfo is None:
        return
    _intrinsics = pyrealsense2.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    _intrinsics.model  = pyrealsense2.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.D]
    depth_array = np.array(depth_map, dtype=np.float32)
    out = []
    for i in points:
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, i, depth_array[[i[1]], [i[0]]] / 1000)
        out.append((result[0], -result[1], -result[2]))
    return out
        
def rotationMatrixToEulerAngles(R):
    ''' 
        Returns Euler angles from matrix x and z swapped
    '''
    from math import atan2, sqrt 
    sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(-R[2, 0], sy)
        z = atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

class Marker:
    def __init__(self, id, corners, name = ""):
        self.id = id
        self.name = name
        self.corners = corners.copy()
        M = cv2.moments(corners)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        self.center = (cX, cY)
        
    def __str__(self):
        return "ID: %x" % (self.id)

def generate_poses_list():
        ms = 0.063 # with space
        poses_list = []
        z_offset = -0.006
        y_offset = 0.0915
        for z in range(4):
            for y in range(3):
                x = 0
                id = 3 * z + y
                poses_list.append((id, x, -y * ms + y_offset, -z * ms + z_offset))
        return np.asarray(poses_list, dtype=np.float64)

class ArucoDetector:
    def __init__(self, ):
        self.filename = rospy.get_param("~calib_path")
        # Subscribe to realsense's topics
        rospy.Subscriber("/camera/color/image_raw", Image, callback=self._image_process, queue_size=2)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self._cam_info_cb)
        self.cv_bridge = CvBridge()

        # advertise topics
        self.vis_pub = rospy.Publisher("/config_pos_node/vis_img", Image, queue_size=10)
        self.header = None
        self.cam_info = None
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.detector_params = aruco.DetectorParameters()
        self.obj_points = generate_poses_list()
        # ROS TF stuff
        self.origin_tf_name = "base"
        self.ee_tf_name = "tool0_controller"
        self.camera_tf_name = "aligned_depth_camera"
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def _cam_info_cb(self, msg):
        self.cam_info = {
            "mtx": np.float32(msg.K).reshape((3,3)),
            "dst": np.float32(msg.D)
        }

    def _get_tf(self, header: Header,
                position: Tuple[float, float, float],
                euler: Tuple[float, float, float]):
        ee_cam_tf = TransformStamped()
        ee_cam_tf.header.stamp = header.stamp
        ee_cam_tf.header.frame_id = self.ee_tf_name
        ee_cam_tf.child_frame_id = self.camera_tf_name
        ee_cam_tf.transform.translation = Vector3(*position)
        camera_ang = np.array(euler)
        q = tf_conversions.transformations.quaternion_from_euler(*camera_ang)
        ee_cam_tf.transform.rotation = Quaternion(*q)

        self.tf_broadcaster.sendTransform(ee_cam_tf)
        try:
            camera_pos = self.tfBuffer.lookup_transform(self.origin_tf_name, self.camera_tf_name, rospy.Time())
            return camera_pos.transform
        except (tf2_ros.ExtrapolationException, tf2_ros.LookupException) as e:
            rospy.logwarn(f'TF lookup failed: "{e}"')
            return None
    
    def _save_tf(self, tf):
        pos = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        ang = np.array([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])

        cv_file = cv2.FileStorage(self.filename, cv2.FILE_STORAGE_WRITE)
        cv_file.writeComment("Date: " + str(datetime.now()))
        cv_file.write("camera_pos", pos)
        cv_file.write("camera_ang", ang)
        cv_file.release()
        rospy.loginfo(f"Calibration saved to {self.filename}")

    def _image_process(self, rgb_msg):
        cv_bridge = self.cv_bridge
        header = deepcopy(rgb_msg.header)
        color_img = cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        # Detecting markers
        markers = self._find_markers(color_img)
        font = cv2.FONT_HERSHEY_PLAIN
        for i, m in enumerate(markers):
            aruco.drawDetectedMarkers(color_img, [m.corners])
            cv2.putText(color_img, str(m.name), m.center, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Estimate camera posistion
        sorted_markers = sorted(markers, key = lambda m : m.id)
        is_recognized = not self.cam_info is None and set([m.id[0] for m in sorted_markers]) == set(static_markers_sked)
        if is_recognized:
            img_points = np.array([m.corners[0][0] for m in sorted_markers], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(self.obj_points[:, 1:].astype('float32'), img_points.astype('float32'), self.cam_info['mtx'], self.cam_info['dst'])
            rotM = cv2.Rodrigues(rvec)[0]
            camPosA = rotM.T
            camPos = -np.matrix(rotM).T * np.matrix(tvec)
            cam_pos = camPos.A1
            cam_ang = rotationMatrixToEulerAngles(camPosA)
            cv2.drawFrameAxes(color_img, self.cam_info['mtx'], self.cam_info['dst'],  rvec, tvec, 0.1)
            cv2.putText(color_img, "Cam pos: X={0:3.4f} Y={1:3.4f} Z={2:3.4f}" .format(*cam_pos), (5, 30), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(color_img, "Cam ang: A={0:3.4f} B={1:3.4f} C={2:3.4f}" .format(*np.rad2deg(cam_ang)), (5, 60), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            rospy.logwarn_throttle_identical(2, "Non-compliance with the required number of static markers or camera info have been not received")
            cv2.putText(color_img, "Cam pos: not estimated", (5, 30), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(color_img, "Cam ang: not estimated", (5, 60), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Color", color_img)
        tf = self._get_tf(header, cam_pos, cam_ang) if is_recognized else None
        if cv2.waitKey(1) == ord("s") and not tf is None:
            self._save_tf(tf)
        self.vis_pub.publish(cv_bridge.cv2_to_imgmsg(color_img, encoding="bgr8"))

    def _find_markers(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_markers = []
        corners, ids, _rejected = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.detector_params)
        if ids is None:
            rospy.logwarn_throttle_identical(2, "Don't see any markers")
        else:
            for i in range(len(ids)):
                detected_markers.append(Marker(ids[i], corners[i], ids[i][0]))
        return list(filter(lambda x: x.id in static_markers_sked, detected_markers))
    

if __name__ == '__main__':
    rospy.init_node("uoais")
    node = ArucoDetector()
    rospy.spin()
