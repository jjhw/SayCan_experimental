#! /usr/bin/env python3

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np

from classes import Circle
from utils import DrawingUtils

class PlateDetector:
    colors = {
    "yellow": ((0, 190, 0), (30, 255, 255)),
    "blue": ((60, 60, 0), (110, 255, 255)),
    "green": ((30, 60, 0), (60, 255, 255)),
    }
    def __init__(self, min_radius=110,
                    max_radius=200,
                    min_solidity=0.7,
                    erode_iterations=5,
                    dilate_iterations=5
                    ):
        """Initializes the plate detector
        @param min_radius: The minimum radius of a plate
        @param max_radius: The maximum radius of a plate
        @param min_solidity: The minimum solidity of a plate
        @param erode_iterations: The number of iterations to erode the mask
        @param dilate_iterations: The number of iterations to dilate the mask
        """
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._min_solidity = min_solidity
        self._erode_iterations = erode_iterations
        self._dilate_iterations = dilate_iterations

    def detect(self, image: np.ndarray) -> List[Circle]:
        """Detects plates in the given image and returns a list of Circle objects
        @param image: The image color BGR image to detect plates in
        @return: A list of Circle objects
        """
        filtered_plates = []
        for color_name, mask in zip(self.colors.keys(), 
                                    self._get_color_mask(image, self.colors.values())):
            mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=self._erode_iterations)
            mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=self._dilate_iterations)
            circles = self._find_circles(mask)
            circles = filter(lambda c: self._min_radius < c['radius'] < self._max_radius 
                             and c['solidity'] > self._min_solidity, circles)
            for circle in circles:
                filtered_plates.append(Circle(name = "{} plate".format(color_name),
                                                mask = circle['mask'],
                                                score = circle['solidity'],
                                                pos = circle['pos'],
                                                area = circle['area'],
                                                radius = circle['radius'],
                                                additional_names = []))
                
        return filtered_plates

    def _find_circles(self, hsv_ranged):
        """Finds circles in the given image
        @param hsv_ranged: The image to find circles in, should be a binary image
        @return: A list of dictionaries containing the circle information
        """
        cnts = cv2.findContours(hsv_ranged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        circles = []
        for c in cnts:
            (pos, radius) = cv2.minEnclosingCircle(c)
            pos = np.array(pos, dtype=np.int32)
            radius = int(radius)
            mask = np.zeros(hsv_ranged.shape, dtype=np.uint8)
            mask = np.expand_dims(mask, axis=2)
            cv2.drawContours(mask, [c], -1, 255, -1)
            area = cv2.countNonZero(mask)
            circle_area = np.pi * radius**2
            solidity = area / circle_area
            circle = {
                "pos": pos,
                "radius": radius,
                "solidity": solidity,
                "area": area,
                "mask": mask
            }
            circles.append(circle)
        return circles
    
    def _get_color_mask(self, image, color_ranges):
        """Returns a generator of masks for the given image and color ranges
        @param image: The image to get the masks for
        @param color_ranges: A list of color ranges to get the masks for
        @return: A generator of masks
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for color_range in color_ranges:
            mask = cv2.inRange(hsv_image, *color_range)
            yield mask

if __name__ == "__main__":
    ac.init_node("alpaca_segmentation_example")
    plate_detector = PlateDetector()
    while not ac.is_shutdown():
        if ac.is_image_ready():
            color, depth_map, _camera_info = ac.pop_image()
            plates = plate_detector.detect(color)
            color = DrawingUtils.draw_plates(color, plates)
            cv2.imshow("color", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            sleep(0.01)