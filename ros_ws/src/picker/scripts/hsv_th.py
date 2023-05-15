#! /usr/bin/env python3

import alpaca_connector as ac
from time import sleep
import cv2


def get_sliders():
    h_min = cv2.getTrackbarPos("h_min", "hsv")
    h_max = cv2.getTrackbarPos("h_max", "hsv")
    s_min = cv2.getTrackbarPos("s_min", "hsv")
    s_max = cv2.getTrackbarPos("s_max", "hsv")
    v_min = cv2.getTrackbarPos("v_min", "hsv")
    v_max = cv2.getTrackbarPos("v_max", "hsv")
    return (h_min, s_min, v_min), (h_max, s_max, v_max)

if __name__ == "__main__":
    ac.init_node("alpaca_segmentation_example")
    cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
    # resize the window
    cv2.resizeWindow("hsv", 1920, 1080)
    # add hsv sliders
    cv2.createTrackbar("h_min", "hsv", 0, 179, lambda x: None)
    cv2.createTrackbar("s_min", "hsv", 0, 255, lambda x: None)
    cv2.createTrackbar("v_min", "hsv", 0, 255, lambda x: None)
    cv2.createTrackbar("h_max", "hsv", 179, 179, lambda x: None)
    cv2.createTrackbar("s_max", "hsv", 255, 255, lambda x: None)
    cv2.createTrackbar("v_max", "hsv", 255, 255, lambda x: None)
    while not ac.is_shutdown():
        if ac.is_image_ready():
            color, depth_map, _camera_info = ac.pop_image()
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            hsv_min, hsv_max = get_sliders()
            print("HSV: ", hsv_min, hsv_max)
            mask = cv2.inRange(hsv, hsv_min, hsv_max)
            cv2.imshow("hsv", mask)
            cv2.imshow("color", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            sleep(0.01)