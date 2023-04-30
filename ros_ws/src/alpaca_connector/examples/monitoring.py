#! /usr/bin/env python3

import alpaca_connector as ac
from time import sleep

ac.init_node("alpaca_connector_monitoring_example")
print("\033[H\033[0J", end="")
print("actual info:")
while not ac.is_shutdown():
    print("\033[2;0H\033[0J", end="")
    print(ac.pose_to_camera())
    sleep(0.01)