#! /usr/bin/env python3

from typing import Any
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

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
        self.shape = self.image.shape
    
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

    def __call__(self):
        return self.image
    

def apply_filters_on_masks(masks):
    # filter out masks with area < 25000
    masks = list(filter(lambda mask: mask['area'] < 25000, masks))
    
    # filter out masks with area / rotated rect area < 0.7
    for mask in masks:
        
        mask['rotated_rect_mask'] = np.zeros(mask['segmentation'].shape, dtype=np.uint8)
        rect = cv2.minAreaRect(np.argwhere(mask['segmentation'] == 1))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(mask['rotated_rect_mask'], [box], 0, 1, -1)
        mask['rotated_rect_area'] = np.sum(mask['rotated_rect_mask'])
    masks = list(filter(lambda mask: mask['area'] / mask['rotated_rect_area'] > 0.7, masks))

    # filter out masks with aspect ratio < 3
    def aspect_ratio_filter(mask):
        x1, y1, w, h = mask['bbox']
        h, w = max(h, w), min(h, w)
        return h / w < 3
    masks = filter(aspect_ratio_filter, masks)

    masks = sorted(masks, key=lambda mask: mask['area'], reverse=False)  
    for mask in masks:
        for other_mask in masks:
            if other_mask is mask:
                continue
            if np.sum(np.logical_and(mask['segmentation'], other_mask['segmentation'])) / np.sum(mask['segmentation']) > 0.8:
                if mask['area'] / other_mask['area'] > 0.05:
                    mask['child'] = True
                    break
            mask['child'] = False
    masks = filter(lambda mask: not mask['child'], masks)

    return masks

def build_masks_tree(masks):
    '''
    Build a tree from masks, where each node is a mask and its children are the masks that are inside it.
    @param masks: the masks to build the tree from
    @return: the root node of the tree
    '''
    root = None
    for mask in masks:
        if root is None:
            root = mask
            continue
        if root['bbox'][0] > mask['bbox'][0] and root['bbox'][1] > mask['bbox'][1] and root['bbox'][2] < mask['bbox'][2] and root['bbox'][3] < mask['bbox'][3]:
            root = mask
    return root

if __name__ == "__main__":
    ac.init_node("alpaca_segmentation_example")
    sam = sam_model_registry["vit_h"](checkpoint="/workspace/weights/sam_vit_h_4b8939.pth").to("cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=3000,
    )
    mask_generator = SamAutomaticMaskGenerator(model=sam)
    predictor = SamPredictor(sam)
    while not ac.is_shutdown():
        if ac.is_image_ready():
            color, depth_map, _camera_info = ac.pop_image()
            crop = CroppedImage(color, [400, 0, 1650, 1080])
            cv2.imshow("color", crop())
            predictor.set_image(crop())
            masks = mask_generator.generate(crop())
            masks = apply_filters_on_masks(masks)
            for i, mask in enumerate(masks):
                # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])   
                # draw mask on color image
                # color[mask['segmentation']] = np.array([0, 255, 0])
                # draw bbox on color image, bbox: [x1, y1, w, h]
                cv2.rectangle(crop(), (mask['bbox'][0], mask['bbox'][1]), (mask['bbox'][0] + mask['bbox'][2], mask['bbox'][1] + mask['bbox'][3]), (0, 0, 255), 2)
                # draw point coords on color image
                rgb_mask = np.zeros(crop().shape, dtype=np.uint8)
                rgb_mask[mask['segmentation']] = np.array([255, 255, 0])
                cv2.addWeighted(crop(), 1, rgb_mask, 0.3, 0, crop())  
                cv2.putText(crop(), str(i), (mask['bbox'][0], mask['bbox'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # put text on color image, text: area/rotated_rect_area
                # cv2.putText(crop(), str(mask['area'] / mask['rotated_rect_area']), (mask['bbox'][0], mask['bbox'][1] + mask['bbox'][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # draw point coords on color image
                # draw crop box on color image, crop_box: [x1, y1, w, h]

            # masks, _, _ = predictor.predict("red block")
            cv2.imshow("color", crop.uncropped())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            sleep(0.01)