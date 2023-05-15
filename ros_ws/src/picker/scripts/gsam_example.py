#! /usr/bin/env python3

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict
import alpaca_connector as ac
from time import sleep
import cv2
import numpy as np
from PIL import Image
from segment_anything import build_sam, SamPredictor 
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict 
from groundingdino.util.inference import predict as gd_predict

import torch
from huggingface_hub import hf_hub_download
from rospy import loginfo, logerr, logwarn

from utils import CroppedImage, DrawingUtils

from classes import Box

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

class GSAMDetector:
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                gd_ckpt_repo_id = "ShilongLiu/GroundingDINO",
                gd_ckpt_filenmae = "groundingdino_swinb_cogcoor.pth",
                gd_ckpt_config_filename = "GroundingDINO_SwinB.cfg.py",
                sam_checkpoint="/workspace/weights/sam_vit_h_4b8939.pth",
                text_threshold = 0.25,
                box_threshold = 0.3
                ):
        self._device = device
        self._load_models(gd_ckpt_repo_id, gd_ckpt_filenmae, gd_ckpt_config_filename, sam_checkpoint)
        self._sam_predictor = SamPredictor(self._sam_model)
        self._text_threshold = text_threshold
        self._box_threshold = box_threshold
    
    def _load_models(self, gd_ckpt_repo_id,
                gd_ckpt_filenmae,
                gd_ckpt_config_filename,
                sam_checkpoint
                ):
        self._gd_model = load_model_hf(gd_ckpt_repo_id, gd_ckpt_filenmae, gd_ckpt_config_filename, device=self._device)
        self._sam_model = build_sam(sam_checkpoint).to(self._device)

    def get_items(self, image: np.ndarray, names: List[str]) -> List[Box]:
        """Find items in the image and return a list of Box objects
        @param image: image to search for items, BGR format
        @param names: list of names to search for, e.g. ['apple', 'banana']
        @return: list of Box objects
        """
        items_list = []
        for name in names:
            boxes, phrases, logits = self._gdino_detect(image, name)
            items_dicts = self._segment(image, boxes, phrases, logits)
            for item in items_dicts:
                item_info = self._process_mask(item)
                if item_info is None:
                    continue
                item_obj = Box(name=name, **item_info, **item, additional_names=[])
                items_list.append(item_obj)
        return items_list
    
    def _gdino_detect(self, image, name):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.fromarray(image).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        boxes, logits, phrases = gd_predict(
            model=self._gd_model,
            image=image_transformed, 
            caption=name,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold
        )
        return boxes, phrases, logits
    
    def _segment(self, image, boxes, phrases, logits):
        H, W, _ = image.shape
        self._sam_predictor.set_image(image)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(boxes_xyxy.to(self._device), image.shape[:2])
        items = []
        for bbox, transformed_box, phrase, logit in zip(boxes_xyxy, transformed_boxes, phrases, logits):
            try:
                masks_predicted, _, _ = self._sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_box.unsqueeze(0),
                        multimask_output = False,
                        )
            except RuntimeError as e:
                logwarn(f"RuntimeError: {e}")
            else:
                mask = np.logical_or.reduce(masks_predicted.cpu().numpy(), axis=0)
                mask = mask.reshape(H, W, 1)
                mask = np.array(mask * 255, dtype=np.uint8)
                bbox =  np.int0(bbox.unsqueeze(0).cpu().numpy())[0]
                item = {
                    'mask': mask,
                    'bbox': bbox,
                    # 'phrase': phrase,
                    'score': logit
                }
                items.append(item)
        return items
            
    def _process_mask(self, item: dict):
        contours, _ = cv2.findContours(item['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return
        contour = contours[0]
        bbox = item['bbox']
        bbox_area = bbox[2] * bbox[3]
        area = int(cv2.contourArea(contour))
        rotated_rect = cv2.minAreaRect(contour)
        (center, (w, h), angle) = rotated_rect
        angle = np.deg2rad(angle)
        if w == 0 or h == 0:
            return
        rotated_rect_area = int(w * h)
        if w < h:
            angle -= np.pi / 2
        w, h = min(w, h), max(w, h)
        
        info = {
            'bbox_area': bbox_area,
            'pos': np.array(center, dtype=np.int32),
            'area': area,
            'rotated_rect': np.int0(cv2.boxPoints(rotated_rect)),
            'rotated_rect_area': rotated_rect_area,
            'width': w,
            'height': h,
            'angle': angle
        }
        return info
    
    
    
    @staticmethod
    def filter_same_items(items: List[Box], dst_threshold: int = 150):
        for item in items:
            if item.score == 0:
                continue
            for other in items:
                if other is item:
                    continue
                if other.score == 0:
                    continue
                if np.linalg.norm(item.pos - other.pos) < dst_threshold:
                    if item.score > other.score:
                        other.score = 0
                    else:
                        item.score = 0
        return list(filter(lambda x: x.score, items))

names_to_detect = ["liquid soap", "fish", "cup", "screwdriver", "cream", "book", "fork", "spoon"]

if __name__ == "__main__":
    ac.init_node("alpaca_gsam_example")
    detector = GSAMDetector()
    while not ac.is_shutdown():
        if ac.is_image_ready():
            color, depth_map, _camera_info = ac.pop_image()
            crop = CroppedImage(color, [400, 0, 1650, 1080])
            items = detector.get_items(crop(), names_to_detect)
            items = GSAMDetector.filter_same_items(items)
            masked_frame = crop().copy()
            for item in items:
                masked_frame = DrawingUtils.draw_box_info(masked_frame, item)
            crop.image = masked_frame
            cv2.imshow("items_detected", crop.uncropped())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            sleep(0.01)