from copy import deepcopy
from dataclasses import dataclass
import rospy
import actionlib
import threading
import tf_conversions
from alpaca_connector.ros_types import *
from typing import Callable, List, Tuple, Union
import numpy as np
import cv2
import time

Pos3D = Tuple[float, float, float]

node_initialized_flag = False
gripper_initialized_flag = False
thread = None

_pose_to_camera = Pose()
gripper_open_srv = None
gripper_close_srv = None

@dataclass
class Point6D:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def __init__(self, x:float = 0, y:float = 0, z:float = 0, roll:float = 0, pitch:float = 0, yaw:float = 0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def from_pose(self, pose: Pose):
        self.x = pose.position.x
        self.y = pose.position.y
        self.z = pose.position.z
        self.roll, self.pitch, self.yaw = tf_conversions.transformations.euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        return self
    
    def to_pose(self) -> Pose:
        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = self.z
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = tf_conversions.transformations.quaternion_from_euler(self.roll, self.pitch, self.yaw)
        return pose

    def __str__(self):
        return "Point6D:\n\tx: {:.6f}...\n\ty: {:.6f}...\n\tz: {:.6f}...\n\troll: {:.6f}...\n\tpitch: {:.6f}...\n\tyaw: {:.6f}...".format(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
    
    def __repr__(self):
        return str(self)
    
    def __iter__(self):
        return iter([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])
    
    def __getitem__(self, key):
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw][key]
    
    def __setitem__(self, key, value):
        # TODO: check if this does work
        # if key == 0:
        #     self.x = value
        # elif key == 1:
        #     self.y = value
        # elif key == 2:
        #     self.z = value
        # elif key == 3:
        #     self.roll = value
        # elif key == 4:
        #     self.pitch = value
        # elif key == 5:
        #     self.yaw = value
        # else:
        #     raise IndexError("Point6D index out of range")
        try:
            [self.x, self.y, self.z, self.roll, self.pitch, self.yaw][key] = value
        except IndexError:
            raise IndexError("Point6D index out of range")
        
    def __len__(self):
        return 6
    
    def __eq__(self, other):
        if isinstance(other, Point6D):
            return all([i == j for i, j in zip(self, other)])
        else:
            return False
        
    def __ne__(self, other):
        return not self.__eq__(other)

def _node_initialized(func):
    def wrapper(*args, **kwargs):
        global node_initialized_flag
        if not node_initialized_flag:
            raise Exception("AlpacaConnector not initialized!\nCall init_node() first!")
        return func(*args, **kwargs)
    return wrapper

def _gripper_initialized(func):
    def wrapper(*args, **kwargs):
        global gripper_initialized_flag
        if not gripper_initialized_flag:
            raise Exception("Gripper initialization was not successful!\nRerun node initialization!")
        return func(*args, **kwargs)
    return wrapper

def init_node(node_name:str = "alpaca_connector"):
    global node_initialized_flag
    if node_initialized_flag:
        raise Exception("AlpacaConnector already initialized!")
    rospy.init_node(node_name)
    node_initialized_flag = True
    _subscribe()
    _action_clients_init()
    _gripper_srv_proxies_init()
    _camera_init()
    _run()


@_node_initialized
def _run():
    global thread
    if thread is not None:
        if thread.is_alive():
            raise Exception("AlpacaConnector already running!")
        else:
            thread.join()
    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()

def _loop():
    global node_initialized_flag, gripper_initialized_flag
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()
    node_initialized_flag = False
    gripper_initialized_flag = False
    exit(0)
    
@_node_initialized
def shutdown():
    global thread, node_initialized_flag, gripper_initialized_flag
    rospy.signal_shutdown("AlpacaConnector shutdown")
    if thread is not None:
        thread.join()
    node_initialized_flag = False
    gripper_initialized_flag = False

@_node_initialized
def _subscribe():
    global pose_to_camera_sub
    #subscribe to topics
    pose_to_camera_sub = rospy.Subscriber("/alpaca/pose_to_camera", Pose, _pose_to_camera_callback)

@_node_initialized
def _action_clients_init():
    global move_by_camera_ac
    #init action clients
    move_by_camera_ac = actionlib.SimpleActionClient("/alpaca/move_by_camera", MovePointsAction)
    #wait for action servers
    move_by_camera_ac.wait_for_server(timeout=rospy.Duration(5))

@_node_initialized
def _gripper_srv_proxies_init():
    global gripper_open_srv, gripper_close_srv, gripper_initialized_flag
    #init service proxies
    open_service_name = "/alpaca/gripper/open"
    close_service_name = "/alpaca/gripper/close"
    #wait for services
    try:
        for srv_name in [open_service_name, close_service_name]:
            rospy.wait_for_service(srv_name, timeout=5)
    except rospy.ROSException as e:
        rospy.logerr("gripper services not found: {}".format(e))
    else:
        gripper_open_srv = rospy.ServiceProxy(open_service_name, Trigger)
        gripper_close_srv = rospy.ServiceProxy(close_service_name, Trigger)
        gripper_initialized_flag = True

def _pose_to_camera_callback(msg):
    global _pose_to_camera
    _pose_to_camera = msg
    
def pose_to_camera() -> Point6D:
    '''Get pose to camera
    :return: Point6D
    '''
    global _pose_to_camera
    point = Point6D().from_pose(_pose_to_camera)
    return point

@_node_initialized
@_gripper_initialized
def gripper(state:bool):
    '''Move gripper to open or close position
    :param state: True for grasp, False for release
    '''
    srv = gripper_close_srv if state else gripper_open_srv
    try:
        ret = srv()
        ret = ret.success
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        ret = False
    return ret
        

@_node_initialized
def move_by_camera(points:List[Point6D]) -> MovePointsResult:
    assert isinstance(points, list) or isinstance(points, tuple), "points must be a list or tuple"
    assert all([isinstance(point, Point6D) for point in points]), "points must be a list or tuple of Point6D"
    assert len(points) > 0, "points must not be empty"
    goal = MovePointsGoal()
    for point in points:
        pose = point.to_pose()
        goal.poses.append(pose)
    move_by_camera_ac.send_goal(goal, feedback_cb=lambda x: rospy.loginfo(f"move by camera progress: {x}"))
    move_by_camera_ac.wait_for_result()
    return move_by_camera_ac.get_result().success.data

on_shutdown = rospy.on_shutdown
is_shutdown = rospy.is_shutdown

_color_buffer = {}
_depth_buffer = {}
_color_msg = None
_depth_msg = None
_new_msg = False
_cv_bridge = CvBridge()
_camera_info = None

def _camera_init():
    global _color_buffer, _depth_buffer, _color_msg, _depth_msg, _new_msg, _cv_bridge, _camera_info
    rospy.Subscriber("/camera/color/image_raw", Image, callback=_rs_message_cb, callback_args={"type": "color"}, queue_size=2)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, callback=_rs_message_cb, callback_args={"type": "depth"}, queue_size=2)
    rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, _cam_info_cb)
    rospy.loginfo("subscribed to camera topics")
    

def _rs_message_cb(msg, args):
    global _color_buffer, _depth_buffer, _color_msg, _depth_msg, _new_msg, _cv_bridge
    timestamp = msg.header.stamp
    if args["type"] == "color":
        if not timestamp in _depth_buffer.keys():
            _clear_buffer()
            _color_buffer[timestamp] = msg
            return
        _color_buffer[timestamp] = msg
    elif args["type"] == "depth":
        if not timestamp in _color_buffer.keys():
            _clear_buffer()
            _depth_buffer[timestamp] = msg
            return
        _depth_buffer[timestamp] = msg
    else:
        rospy.logerr(f"Unknown message type: {args['type']}")
        return
    _new_msg = True
    _color_msg = _color_buffer[timestamp]
    _depth_msg = _depth_buffer[timestamp]

def _cam_info_cb(msg):
    # Save the camera intrinsic parameters
    global _camera_info
    _camera_info = msg

def _clear_buffer(max_len=10):
    global _color_buffer, _depth_buffer
    if len(_color_buffer) > max_len or len(_depth_buffer) > max_len:
        _color_buffer = {}
        _depth_buffer = {}

@_node_initialized
def pop_image() -> Tuple[np.ndarray, np.ndarray, CameraInfo]:
    '''Get color and depth image from realsense
    :return: (color image, depth image)
    '''
    global _new_msg, _color_msg, _depth_msg, _cv_bridge, _camera_info
    if _camera_info is None:
        raise RuntimeError("Camera info is not ready")
    _new_msg = False
    color_image = _cv_bridge.imgmsg_to_cv2(_color_msg, desired_encoding="bgr8")
    depth_map = _cv_bridge.imgmsg_to_cv2(_depth_msg, desired_encoding="32FC1")
    return color_image, depth_map, _camera_info

@_node_initialized
def wait_for_image(timeout=5) -> Tuple[np.ndarray, np.ndarray, CameraInfo]:
    '''Wait for the latest image to be ready
    '''
    global _new_msg
    start_time = time.time()
    while not _new_msg:
        if time.time() - start_time > timeout and not timeout is None:
            raise RuntimeError("Timeout waiting for image")
        time.sleep(0.1)
    return pop_image()

@_node_initialized
def is_image_ready() -> bool:
    '''Check if the latest image is ready
    :return: True if the latest image is ready, False otherwise
    '''
    global _new_msg, _camera_info
    if _new_msg and _camera_info is not None:
        return True
    return False

def to_real_map(depth_map:np.ndarray, camera_info:CameraInfo=None) -> np.ndarray:
    '''Convert depth image to real world coordinates
    :param depth_map: depth image
    :param camera_info: camera intrinsic parameters (optional)
    :return: real world coordinates
    '''
    global _camera_info
    if camera_info is None:
        camera_info = _camera_info
    assert isinstance(depth_map, np.ndarray), "depth_map must be a numpy array"
    assert isinstance(camera_info, CameraInfo), "camera_info must be a CameraInfo"
    assert len(depth_map.shape) == 2, "depth_map must be a 2D array"
    assert depth_map.shape[0] == camera_info.height and depth_map.shape[1] == camera_info.width, "depth_map and camera_info must have the same size"
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]
    depth_map = depth_map.astype(np.float32)
    depth_map[depth_map == 0] = np.nan
    x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
    x = (x - cx) * depth_map / fx
    y = (y - cy) * depth_map / fy
    z = depth_map
    return np.stack([x, y, z], axis=-1)

def to_real_points(points:List, depth_map:np.ndarray, camera_info:CameraInfo=None) -> List[Point]:
    '''Convert a list of points to real world coordinates
    :param points: list of points
    :param depth_map: depth image
    :param camera_info: camera intrinsic parameters (optional)
    :return: list of points in real world coordinates
    '''
    global _camera_info
    if camera_info is None:
        camera_info = _camera_info
    assert isinstance(points, list), "points must be a list"
    assert isinstance(depth_map, np.ndarray), "depth_map must be a numpy array"
    assert isinstance(camera_info, CameraInfo), "camera_info must be a CameraInfo"
    assert len(depth_map.shape) == 2, "depth_map must be a 2D array"
    assert depth_map.shape[0] == camera_info.height and depth_map.shape[1] == camera_info.width, "depth_map and camera_info must have the same size"
    real_map = to_real_map(depth_map, camera_info)
    real_points = []
    for point in points:
        if point[0] < 0 or point[0] >= camera_info.width or point[1] < 0 or point[1] >= camera_info.height:
            raise ValueError(f"Point {point} is out of camera's view")
        x = point[0]
        y = point[1]
        z = real_map[int(y), int(x), 2]
        real_points.append((x, y, z))
    return real_points
