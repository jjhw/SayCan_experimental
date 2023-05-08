#! /usr/bin/env python3

import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np

depth_map = None


def normalize_depth(depth_map, low, high):
    depth_map = depth_map.astype(float)
    undetectable_mask = depth_map == 0
    depth_map = (depth_map - low) / (high - low)
    # 10, 11 and 12 are magic numbers for clipping
    depth_map[depth_map > 1] = 10
    depth_map[depth_map < 0] = 11
    higher_mask = depth_map == 10
    lower_mask = depth_map == 11
    depth_map = np.clip(depth_map, 0, 1)
    depth_map = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    depth_map[higher_mask] = (255, 255, 255)
    depth_map[lower_mask] = (100, 100, 100)
    depth_map[undetectable_mask] = (0, 0, 0)
    return depth_map

def mouse_callback(event, x, y, flags, param):
    global depth_map
    if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
        # get real world coordinates
        xyz = ac.to_real_points([(x, y)], depth_map)
        print(f"Mouse position: {x}, {y} | Depth: {depth_map[y, x]} | XYZ: {xyz}")

if __name__ == "__main__":
    ac.init_node("alpaca_connector_realsense_example")
    # add callback to imshow window mouse position
    
    cv2.namedWindow("color")
    cv2.setMouseCallback("color", mouse_callback)
    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", mouse_callback)
    while not ac.is_shutdown():
        color, depth_map, _camera_info = ac.wait_for_image()
        depth = normalize_depth(depth_map, 930, 980)
        cv2.imshow("color", color)
        cv2.imshow("depth", depth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break