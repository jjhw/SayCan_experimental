#! /usr/bin/env python3

from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np
from PIL import Image
import os, sys
from collections import namedtuple
from rospy import loginfo, logerr, logwarn
import matplotlib.pyplot as plt

class CroppedImage:
    def __init__(self, image, crop_box):
        '''
        @param image: the image to crop
        @param crop_box: the crop box, [x1, y1, x2, y2]
        '''
        self.image = image
        self._original_shape = image.shape
        self._crop_box = crop_box
        self._crop()

        # public attributes
        self.shape = self.image.shape
        self.crop_box = crop_box.copy()
    
    def _crop(self):
        x1, y1, x2, y2 = self._crop_box
        self.image = self.image[y1:y2, x1:x2]
    
    def uncropped(self, fill_value=[0, 0, 0]):
        '''
        Return the image with the cropped region filled with fill_value
        @param fill_value: the value to fill the cropped region with
        '''
        image = np.full(self._original_shape, fill_value, dtype=np.uint8)
        x1, y1, x2, y2 = self._crop_box
        image[y1:y2, x1:x2] = self.image
        return image
    
    def copy(self):
        return CroppedImage(self.image.copy(), self.crop_box.copy())
                            
    def __call__(self):
        return self.image

def process_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    rotated_rect = cv2.minAreaRect(contour)
    rotated_rect_area = int(rotated_rect[1][0] * rotated_rect[1][1])
    center = np.array(rotated_rect[0], dtype=np.int32)
    area = int(cv2.contourArea(contour))
    rotated_rect = np.int0(cv2.boxPoints(rotated_rect))
    return rotated_rect, rotated_rect_area, center, area

def draw_mask(mask, image, color:np.ndarray=None):
    if color is None:
        color = np.random.uniform(0, 1, size=3)
    rgb_mask = np.array([mask.copy() for _ in range(3)])
    colored_mask = np.concatenate(rgb_mask, axis=-1) * color
    colored_mask = np.array(colored_mask, dtype=np.uint8)
    image = cv2.addWeighted(image, 1, colored_mask, 1, 0)
    return image

height_map_path = "/workspace/config/height_map.npy"
plain_offset = 1 # [mm]

if __name__ == "__main__":
    ac.init_node("alpaca_segmentation_example")
    try:
        height_map = np.load(height_map_path)
    except FileNotFoundError:
        logerr("Height map not found!")
        exit(1)
    while not ac.is_shutdown():
        if ac.is_image_ready():
            color, depth_map, _camera_info = ac.pop_image()
            height_mask = np.zeros_like(depth_map)
            height_mask[height_map - plain_offset > depth_map] = 255
            height_mask[depth_map > 1000] = 0
            height_mask[depth_map == 0] = 0
            height_mask = cv2.erode(height_mask, np.ones((5, 5), dtype=np.uint8), iterations=3)
            height_mask = cv2.dilate(height_mask, np.ones((5, 5), dtype=np.uint8), iterations=5)
            
            # add weighted mask
            height_mask = np.expand_dims(height_mask, axis=2)
            height_mask = np.array([height_mask.copy() for _ in range(3)])
            height_mask = np.concatenate(height_mask, axis=-1)
            height_mask = np.array(height_mask, dtype=np.uint8)
            color = cv2.addWeighted(color, 1, ~height_mask, 0.5, 0)
            # crop = CroppedImage(color, [400, 0, 1650, 1080])

            cv2.imshow("color", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            sleep(0.01)