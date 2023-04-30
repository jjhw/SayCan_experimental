#! /usr/bin/env python3

import alpaca_connector as ac
from time import sleep

ac.init_node("alpaca_connector_gripper_example")
while not ac.is_shutdown():
    print("grasping object")
    print(ac.gripper(True))
    sleep(3)
    print("releasing object")
    print(ac.gripper(False))
    sleep(3)