from copy import deepcopy
from dataclasses import dataclass
import rospy
import actionlib
import threading
import tf_conversions
from alpaca_connector.ros_types import *
from typing import Callable, List, Tuple, Union

Pos3D = Tuple[float, float, float]

node_initialized_flag = False
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
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value
        elif key == 3:
            self.roll = value
        elif key == 4:
            self.pitch = value
        elif key == 5:
            self.yaw = value
        else:
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

def init_node(node_name:str = "alpaca_connector"):
    global node_initialized_flag
    if node_initialized_flag:
        raise Exception("AlpacaConnector already initialized!")
    rospy.init_node(node_name)
    node_initialized_flag = True
    _subscribe()
    _action_clients_init()
    _srv_proxies_init()
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
    global node_initialized_flag
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        rate.sleep()
    node_initialized_flag = False
    exit(0)
    
@_node_initialized
def shutdown():
    global thread, node_initialized_flag
    rospy.signal_shutdown("AlpacaConnector shutdown")
    if thread is not None:
        thread.join()
    node_initialized_flag = False

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
def _srv_proxies_init():
    global gripper_open_srv, gripper_close_srv
    #init service proxies
    open_service_name = "/alpaca/gripper/open"
    close_service_name = "/alpaca/gripper/close"
    gripper_open_srv = rospy.ServiceProxy(open_service_name, Trigger)
    gripper_close_srv = rospy.ServiceProxy(close_service_name, Trigger)
    #wait for services
    for srv_name in [open_service_name, close_service_name]:
        rospy.wait_for_service(srv_name, timeout=5)

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