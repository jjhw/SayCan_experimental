#! /usr/bin/env python3

import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import seaborn as sns
import pandas as pd
import scipy.interpolate as interp

point = namedtuple("point", ["x", "y", "depth"])
from scipy.ndimage import morphology

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
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
        # get real world coordinates
        if cv2.EVENT_LBUTTONDOWN:
            param.append(point(x, y, depth_map[y, x]))
        xyz = ac.to_real_points([(x, y)], depth_map)
        print(f"Mouse position: {x}, {y} | Depth: {depth_map[y, x]} | XYZ: {xyz}")


def fill_nan_values(height_map):
    while np.isnan(height_map).any():

        # Iterate through the height map by step (10, 10)
        for i, j in  np.ndindex(height_map.shape[0] // 10, height_map.shape[1] // 10):
            x, y = i * 10, j * 10
            # Create a 20x20 core centered at (i, j)
            core = height_map[x - 10:x + 10, y - 10:y + 10]

            # Find the nearest non-NaN value to (i, j)
            non_nan_mask = ~np.isnan(core)
            nan_mask = np.isnan(core)
            if non_nan_mask.any() and nan_mask.any():
                nearest_non_nan = core[non_nan_mask].mean()
            else:
                continue

            # Fill all NaN values in the core with the nearest non-NaN value
            height_map[ x - 10:x + 10, y - 10:y + 10][nan_mask] = nearest_non_nan

    return height_map

def gaussian_filter(height_map):
    # Create a Gaussian filter
    kernel = cv2.getGaussianKernel(20, None)
    kernel = kernel * kernel.T

    # Apply the filter to the height map
    height_map = cv2.filter2D(height_map, -1, kernel)
    return height_map

def generate_depth_gradient(points):
    # define the size of the grid
    width, height = 1920, 1080
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # define some points with defined height values
    points = [(p.x, p.y, p.depth) for p in points]
    points = np.array(points)

    interp_func =interp.LinearNDInterpolator(points[:, :2], points[:, 2])
    height_map = interp_func(np.column_stack((x.ravel(), y.ravel()))).reshape((height, width))

    # # fill the NaN values outside of the defined area with linear interpolation
    # is_outside = np.isnan(height_map)
    # height_map[is_outside] = np.interp(np.flatnonzero(is_outside), np.flatnonzero(~is_outside), height_map[~is_outside])
    height_map = fill_nan_values(height_map)
    # apply gaussian filter to smooth the height map
    for _ in range(10):
        height_map = gaussian_filter(height_map)
    return height_map


plain_offset = 5 # [mm]
height_map_path = "/workspace/config/height_map.npy"
if __name__ == "__main__":
    ac.init_node("alpaca_connector_realsense_example")
    # add callback to imshow window mouse position
    clicked_points = []
    cv2.namedWindow("color")
    cv2.setMouseCallback("color", mouse_callback, param=clicked_points)
    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", mouse_callback, param=clicked_points)
    try:
        # load height map from file
        height_map = np.load(height_map_path)
    except:
        height_map = None
    # height_map = None
    while not ac.is_shutdown():
        color, depth_map, _camera_info = ac.wait_for_image()
        if height_map is not None:
            # generate height mask based on height map and depth map
            height_mask = np.zeros_like(depth_map)
            height_mask[height_map - plain_offset > depth_map] = 255
            height_mask[depth_map > 1000] = 0
            height_mask[depth_map == 0] = 0
            # apply height mask to color image
            color_masked = color.copy()
            color_masked[height_mask == 0] = 255
            # apply mask to depth image
            depth_masked = depth_map.copy()
            depth_masked[height_mask == 0] = 2000
            depth_masked = normalize_depth(depth_masked, 900, np.max(height_map))
            cv2.imshow("color", color_masked)
            cv2.imshow("depth", depth_masked)
            # normalize height map
            height_map_norm = normalize_depth(height_map, 900, np.max(height_map))
            cv2.imshow("height_map", height_map_norm)
        else:
            depth = normalize_depth(depth_map, 900, 1000)
            cv2.imshow("color", color)
            cv2.imshow("depth", depth)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(clicked_points)
            height_map = generate_depth_gradient(clicked_points)
            # save height map as numpy array
            np.save(height_map_path, height_map)
            # np.save("clicked_points.npy", clicked_points)
            # break