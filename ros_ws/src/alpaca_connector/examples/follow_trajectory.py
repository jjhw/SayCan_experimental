#! /usr/bin/env python3

import alpaca_connector as ac
from time import sleep
import numpy as np

ac.init_node("alpaca_connector_follow_trajectory_example")
# while not ac.is_shutdown():
trajectory = [
    ac.Point6D(0, 0, 0.6, np.pi, 0, 0),
    ac.Point6D(0, 0.1, 0.8, np.pi, 0, np.pi/3),
    ac.Point6D(0, -0.1, 0.5, np.pi, 0, -np.pi/3),
]
print(ac.move_by_camera(trajectory))