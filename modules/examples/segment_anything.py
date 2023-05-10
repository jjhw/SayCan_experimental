#! /usr/bin/env python3 

from vild import Vild, FLAGS
import alpaca_connector as ac
import cv2
import numpy as np

category_names = ['red block', 'green block', 'blue block', 'yellow block']
category_name_string = ";".join(category_names)
prompt_swaps = []

if __name__ == "__main__":
    ac.init_node("vild_example_node")
    vild_detector = Vild(FLAGS,
                            max_boxes_to_draw = 8,
                            nms_threshold = 0.4,
                            min_rpn_score_thresh = 0.4,
                            min_box_area = 1000,
                            max_box_area = 30000
                            )
    while not ac.is_shutdown():
        color, depth_map, _camera_info = ac.wait_for_image()
        found_objects, img = vild_detector.process(color, category_names, prompt_swaps)
        print(found_objects)
        if not img is None:
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break