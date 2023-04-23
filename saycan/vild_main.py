from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch import softmax
from vild_prompt import FLAGS, build_text_embedding
from vild_viz import paste_instance_masks, display_image, visualize_boxes_and_labels_on_image_array

def nms(dets, scores, thresh, max_dets=1000):
  """Non-maximum suppression.
  Args:
    dets: [N, 4]
    scores: [N,]
    thresh: iou threshold. Float
    max_dets: int.
  """
  y1 = dets[:, 0]
  x1 = dets[:, 1]
  y2 = dets[:, 2]
  x2 = dets[:, 3]

  areas = (x2 - x1) * (y2 - y1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0 and len(keep) < max_dets:
    i = order[0]
    keep.append(i)

    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
    overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

    inds = np.where(overlap <= thresh)[0]
    order = order[inds + 1]
  return keep

class VILD:
    def __init__(self, session, overall_fig_size, numbered_category_indices, clip_model):
        self.session = session
        self.overall_fig_size = overall_fig_size
        self.numbered_category_indices = numbered_category_indices
        self.clip_model = clip_model

    def vild(self, image_path, category_name_string, params, plot_on=True, prompt_swaps=[]):
        #################################################################
        # Preprocessing categories and get params
        for a, b in prompt_swaps:
            category_name_string = category_name_string.replace(a, b)
        category_names = [x.strip() for x in category_name_string.split(";")]
        category_names = ["background"] + category_names
        categories = [{"name": item, "id": idx+1,} for idx, item in enumerate(category_names)]
        category_indices = {cat["id"]: cat for cat in categories}
    
        max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area = params
        fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)


        #################################################################
        # Obtain results and read image
        roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = self.session.run(
            ["RoiBoxes:0", "RoiScores:0", "2ndStageBoxes:0", "2ndStageScoresUnused:0", "BoxOutputs:0", "MaskOutputs:0", "VisualFeatOutputs:0", "ImageInfo:0"],
            feed_dict={"Placeholder:0": [image_path,]})
    
        roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
        # no need to clip the boxes, already done
        roi_scores = np.squeeze(roi_scores, axis=0)

        detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
        scores_unused = np.squeeze(scores_unused, axis=0)
        box_outputs = np.squeeze(box_outputs, axis=0)
        detection_masks = np.squeeze(detection_masks, axis=0)
        visual_features = np.squeeze(visual_features, axis=0)

        image_info = np.squeeze(image_info, axis=0)  # obtain image info
        image_scale = np.tile(image_info[2:3, :], (1, 2))
        image_height = int(image_info[0, 0])
        image_width = int(image_info[0, 1])

        rescaled_detection_boxes = detection_boxes / image_scale # rescale

        # Read image
        image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
        assert image_height == image.shape[0]
        assert image_width == image.shape[1]


        #################################################################
        # Filter boxes

        # Apply non-maximum suppression to detected boxes with nms threshold.
        nmsed_indices = nms(
            detection_boxes,
            roi_scores,
            thresh=nms_threshold
            )

        # Compute RPN box size.
        box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

        # Filter out invalid rois (nmsed rois)
        valid_indices = np.where(
            np.logical_and(
            np.isin(np.arange(len(roi_scores), dtype=np.int), nmsed_indices),
            np.logical_and(
                np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                np.logical_and(
                    roi_scores >= min_rpn_score_thresh,
                    np.logical_and(
                    box_sizes > min_box_area,
                    box_sizes < max_box_area
                    )
                    )
            )    
            )
        )[0]

        detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
        detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
        detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
        detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
        rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]


        #################################################################
        # Compute text embeddings and detection scores, and rank results
        text_features = build_text_embedding(self.clip_model, categories)
    
        raw_scores = detection_visual_feat.dot(text_features.T)
        if FLAGS.use_softmax:
            scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
        else:
            scores_all = raw_scores

        indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
        indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

    
        #################################################################
        # Print found_objects
        found_objects = []
        for a, b in prompt_swaps:
            category_names = [name.replace(b, a) for name in category_names]  # Extra prompt engineering.
        for anno_idx in indices[0:int(rescaled_detection_boxes.shape[0])]:
            scores = scores_all[anno_idx]
            if np.argmax(scores) == 0:
                continue
            found_object = category_names[np.argmax(scores)]
            if found_object == "background":
                continue
            print("Found a", found_object, "with score:", np.max(scores))
            found_objects.append(category_names[np.argmax(scores)])
        if not plot_on:
            return found_objects
    

        #################################################################
        # Plot detected boxes on the input image.
        ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
        processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
        segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

        if len(indices_fg) == 0:
            display_image(np.array(image), size=self.overall_fig_size)
            print("ViLD does not detect anything belong to the given category")

        else:
            image_with_detections = visualize_boxes_and_labels_on_image_array(
            np.array(image),
            rescaled_detection_boxes[indices_fg],
            valid_indices[:max_boxes_to_draw][indices_fg],
            detection_roi_scores[indices_fg],    
            self.numbered_category_indices,
            instance_masks=segmentations[indices_fg],
            use_normalized_coordinates=False,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_rpn_score_thresh,
            skip_scores=False,
            skip_labels=True)

        # plt.figure(figsize=overall_fig_size)
        plt.imshow(image_with_detections)
        # plt.axis("off")
        plt.title("ViLD detected objects and RPN scores.")
        plt.show()

        return found_objects
    
    