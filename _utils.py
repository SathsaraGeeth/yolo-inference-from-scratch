# =============================================================================
# This file contains code derived from multiple sources:
#
# 1. Ultralytics YOLOv5 project
#    - Repository: https://github.com/ultralytics/yolov5.git
#    - Copyright (c) 2020–present Ultralytics LLC
#    - Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
#
# 2. YOLOV5m by Alessandro Mondin (unlicensed)
#    - Repository: https://github.com/AlessandroMondin/YOLOV5m.git
#
# Modifications, integration, and additional functionality by Geeth Sathsara
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero General Public License for more details:
# https://www.gnu.org/licenses/
# =============================================================================

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import torch
from torchvision.ops import nms
import cv2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# ALADDIN'S
def iou_width_height(gt_box, anchors, strided_anchors=True, stride=[8, 16, 32]):
    """
    Parameters:
        gt_box (tensor): width and height of the ground truth box
        anchors (tensor): lists of anchors containing width and height
        strided_anchors (bool): if the anchors are divided by the stride or not
    Returns:
        tensor: Intersection over union between the gt_box and each of the n-anchors
    """
    # boxes 1 (gt_box): shape (2,)
    # boxes 2 (anchors): shape (9,2)
    # intersection shape: (9,)
    anchors /= 640
    if strided_anchors:
        anchors = anchors.reshape(9, 2) * torch.tensor(stride).repeat(6, 1).T.reshape(9, 2)

    intersection = torch.min(gt_box[..., 0], anchors[..., 0]) * torch.min(
        gt_box[..., 1], anchors[..., 1]
    )
    union = (
        gt_box[..., 0] * gt_box[..., 1] + anchors[..., 0] * anchors[..., 1] - intersection
    )
    # intersection/union shape (9,)
    return intersection / union


# ALADDIN'S MODIFIED
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", GIoU=False, eps=1e-7):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        GIoU (bool): if True it computed GIoU loss (https://giou.stanford.edu)
        eps (float): for numerical stability

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    else:  # if not midpoints box coordinates are considered to be in coco format
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    w1, h1, w2, h2 = box1_x2 - box1_x1, box1_y2 - box1_y1, box2_x2 - box2_x1, box2_y2 - box2_y1
    # Intersection area
    inter = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0) * \
            (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU:
        cw = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        c_area = cw * ch + eps  # convex height
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

# found here: https://gist.github.com/cbernecker/1ac2f9d45f28b6a4902ba651e3d4fa91#file-coco_to_yolo-py
def coco_to_yolo(bbox, image_w=640, image_h=640):
    x1, y1, w, h = bbox
    #return [((x1 + w)/2)/image_w, ((y1 + h)/2)/image_h, w/image_w, h/image_h]
    return [((2*x1 + w)/(2*image_w)), ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def coco_to_yolo_tensors(bbox, w0=640, h0=640):
    x1, y1, w, h = np.split(bbox, 4, axis=1)
    #return [((x1 + w)/2)/image_w, ((y1 + h)/2)/image_h, w/image_w, h/image_h]
    return np.concatenate([((2*x1 + w)/(2*w0)), ((2*y1 + h)/(2*h0)), w/w0, h/h0], axis=1)


# rescales bboxes from an image_size to another image_size
"""def rescale_bboxes(bboxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    new_boxes = []
    for bbox in bboxes:
        x = math.floor(bbox[0] * ew/sw * 100)/100
        y = math.floor(bbox[1] * eh/sh * 100)/100
        w = math.floor(bbox[2] * ew/sw * 100)/100
        h = math.floor(bbox[3] * eh/sh * 100)/100
        
        new_boxes.append([x, y, w, h])
    return new_boxes"""


def rescale_bboxes(bboxes, starting_size, ending_size):
    sw, sh = starting_size
    ew, eh = ending_size
    y = np.copy(bboxes)

    y[:, 0:1] = np.floor(bboxes[:, 0:1] * ew / sw * 100)/100
    y[:, 1:2] = np.floor(bboxes[:, 1:2] * eh / sh * 100)/100
    y[:, 2:3] = np.floor(bboxes[:, 2:3] * ew / sw * 100)/100
    y[:, 3:4] = np.floor(bboxes[:, 3:4] * eh / sh * 100)/100

    return y

# ALADDIN'S
def non_max_suppression_aladdin(bboxes, iou_threshold, threshold, box_format="corners", max_detections=300):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    if len(bboxes) > max_detections:
        bboxes = bboxes[:max_detections]

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def non_max_suppression(batch_bboxes, iou_threshold, threshold, max_detections=300, tolist=True):

    """new_bboxes = []
    for box in bboxes:
        if box[1] > threshold:
            box[3] = box[0] + box[3]
            box[2] = box[2] + box[4]
            new_bboxes.append(box)"""

    bboxes_after_nms = []
    for boxes in batch_bboxes:
        boxes = torch.masked_select(boxes, boxes[..., 1:2] > threshold).reshape(-1, 6)

        # from xywh to x1y1x2y2

        boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
        boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:] / 2)
        boxes[..., 5:6] = boxes[..., 5:6] + boxes[..., 3:4]
        boxes[..., 4:5] = boxes[..., 4:5] + boxes[..., 2:3]

        indices = nms(boxes=boxes[..., 2:] + boxes[..., 0:1], scores=boxes[..., 1], iou_threshold=iou_threshold)
        boxes = boxes[indices]

        # sorts boxes by objectness score but it's already done internally by torch metrics's nms
        # _, si = torch.sort(boxes[:, 1], dim=0, descending=True)
        # boxes = boxes[si, :]

        if boxes.shape[0] > max_detections:
            boxes = boxes[:max_detections, :]

        bboxes_after_nms.append(
            boxes.tolist() if tolist else boxes
        )

    return bboxes_after_nms if tolist else torch.cat(bboxes_after_nms, dim=0)




def cells_to_bboxes(predictions, anchors, strides, is_pred=False, to_list=True):
    num_out_layers = len(predictions)
    grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
    anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
        
    all_bboxes = []
    for i in range(num_out_layers):
        bs, naxs, ny, nx, _ = predictions[i].shape
        stride = strides[i]
        grid[i], anchor_grid[i] = make_grids(anchors, naxs, ny=ny, nx=nx, stride=stride, i=i)
        if is_pred:
            # formula here: https://github.com/ultralytics/yolov5/issues/471
            #xy, wh, conf = predictions[i].sigmoid().split((2, 2, 80 + 1), 4)
            layer_prediction = predictions[i].sigmoid()
            obj = layer_prediction[..., 4:5]
            xy = (2 * (layer_prediction[..., 0:2]) + grid[i] - 0.5) * stride
            wh = ((2*layer_prediction[..., 2:4])**2) * anchor_grid[i]
            best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)

        else:
            predictions[i] = predictions[i].to(DEVICE, non_blocking=True)
            obj = predictions[i][..., 4:5]
            xy = (predictions[i][..., 0:2] + grid[i]) * stride
            wh = predictions[i][..., 2:4] * stride
            best_class = predictions[i][..., 5:6]

        scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)

        all_bboxes.append(scale_bboxes)

    return torch.cat(all_bboxes, dim=1).tolist() if to_list else torch.cat(all_bboxes, dim=1)

def make_grids(anchors, naxs, stride, nx=20, ny=20, i=0):

    x_grid = torch.arange(nx)
    x_grid = x_grid.repeat(ny).reshape(ny, nx)

    y_grid = torch.arange(ny).unsqueeze(0)
    y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)

    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)
    anchor_grid = (anchors[i]*stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

    return xy_grid, anchor_grid


def save_predictions(model, loader, folder, epoch, device, filename, num_images=10, labels=COCO):

    print("=> Saving images predictions...")

    if not os.path.exists(path=os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch)}')):
        os.makedirs(os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch)}'))

    path = os.path.join(os.getcwd(), folder, filename, f'EPOCH_{str(epoch)}')
    anchors = model.head.anchors

    model.eval()

    for idx, (images, targets) in enumerate(loader):

        images = images.to(device).float()/255

        if idx < num_images:
            with torch.no_grad():
                out = model(images)

            boxes = cells_to_bboxes(out, anchors, model.head.stride, is_pred=True, list_output=False)
            gt_boxes = cells_to_bboxes(targets, anchors, model.head.stride, is_pred=False, list_output=False)

            # here using different nms_iou_thresh and config_thresh because of
            # https://github.com/ultralytics/yolov5/issues/4464
            boxes = nms(boxes, iou_threshold=0.45, threshold=0.25)[0]
            gt_boxes = nms(gt_boxes, iou_threshold=0.45, threshold=0.7)[0]

            cmap = plt.get_cmap("tab20b")
            class_labels = labels
            colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
            im = np.array(images[0].permute(1, 2, 0).cpu())

            # Create figure and axes
            fig, (ax1, ax2) = plt.subplots(1, 2)
            # Display the image
            ax1.imshow(im)
            ax2.imshow(im)

            # box[0] is x midpoint, box[2] is width
            # box[1] is y midpoint, box[3] is height
            axes = [ax1, ax2]
            # Create a Rectangle patch
            boxes = [gt_boxes, boxes]
            for i in range(2):
                for box in boxes[i]:
                    assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"

                    class_pred = int(box[0])
                    box = box[2:]
                    upper_left_x = max(box[0], 0)
                    upper_left_x = min(upper_left_x, im.shape[1])
                    lower_left_y = max(box[1], 0)
                    lower_left_y = min(lower_left_y, im.shape[0])

                    # print(upper_left_x)
                    # print(lower_left_y)
                    rect = patches.Rectangle(
                        (upper_left_x, lower_left_y),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=1,
                        edgecolor=colors[class_pred],
                        facecolor="none",
                    )
                    # Add the patch to the Axes
                    if i == 0:
                        axes[i].set_title("Ground Truth bboxes")
                    else:
                        axes[i].set_title("Predicted bboxes")
                    axes[i].add_patch(rect)
                    axes[i].text(
                        upper_left_x,
                        lower_left_y,
                        s=class_labels[class_pred],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": colors[class_pred], "pad": 0},
                        fontsize="xx-small"
                    )

            fig.savefig(f'{path}/image_{idx}.png', dpi=300)
            plt.close(fig)
        # if idx > num images
        else:
            break

    model.train()


# def plot_image(image, boxes, labels=COCO):
#     """Plots predicted bounding boxes on the image"""
#     cmap = plt.get_cmap("tab20b")
#     class_labels = labels
#     colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
#     im = np.array(image)

#     # Create figure and axes
#     fig, ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(im)

#     # box[0] is x midpoint, box[2] is width
#     # box[1] is y midpoint, box[3] is height

#     # Create a Rectangle patch
#     for box in boxes:
#         assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
#         class_pred = box[0]
#         bbox = box[2:]

#         # FOR MY_NMS attempts, also rect = patches.Rectangle box[2] becomes box[2] - box[0] and box[3] - box[1]
#         upper_left_x = max(bbox[0], 0)
#         upper_left_x = min(upper_left_x, im.shape[1])
#         lower_left_y = max(bbox[1], 0)
#         lower_left_y = min(lower_left_y, im.shape[0])

#         """upper_left_x = max(box[0] - box[2] / 2, 0)
#         upper_left_x = min(upper_left_x, im.shape[1])
#         lower_left_y = max(box[1] - box[3] / 2, 0)
#         lower_left_y = min(lower_left_y, im.shape[0])"""

#         rect = patches.Rectangle(
#             (upper_left_x, lower_left_y),
#             bbox[2] - bbox[0],
#             bbox[3] - bbox[1],
#             linewidth=2,
#             edgecolor=colors[int(class_pred)],
#             facecolor="none",
#         )
#         # Add the patch to the Axes
#         ax.add_patch(rect)
#         plt.text(
#             upper_left_x,
#             lower_left_y,
#             s=f"{class_labels[int(class_pred)]}: {box[1]:.2f}",
#             color="white",
#             verticalalignment="top",
#             bbox={"color": colors[int(class_pred)], "pad": 0},
#         )
#     plt.show()

from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image, boxes, labels):
    class_labels = labels
    im = np.array(image)
    if im.dtype != np.uint8:
        im = (im * 255).astype(np.uint8)
    if im.ndim == 3 and im.shape[0] == 3:
        im = np.transpose(im, (1, 2, 0))
    pil_img = Image.fromarray(im).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    cmap = plt.get_cmap("tab20b")
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in np.linspace(0, 1, len(class_labels))
    ]
    for box in boxes:
        assert len(box) == 6, "box should contain class, conf, x1, y1, x2, y2"
        class_pred = int(box[0])
        bbox = box[2:]
        x1 = int(max(min(bbox[0], im.shape[1]), 0))
        y1 = int(max(min(bbox[1], im.shape[0]), 0))
        x2 = int(max(min(bbox[2], im.shape[1]), 0))
        y2 = int(max(min(bbox[3], im.shape[0]), 0))
        color = colors[class_pred]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        class_name = class_labels[class_pred]
        confidence = box[1]
        label = f"{class_name}: {confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle(
            [x1, y1 - text_h, x1 + text_w, y1],
            fill=color
        )
        draw.text(
            (x1, y1 - text_h),
            label,
            fill=(255, 255, 255),
            font=font
        )
    display(pil_img)
