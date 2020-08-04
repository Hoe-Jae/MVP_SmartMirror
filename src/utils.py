import cv2
import torch
from collections import Counter

def detect(out, cls_high, cls_before, ban_list=[4,5,7,8,9,10,12]):
    
    if out[0] is not None:
        
        x1, y1, x2, y2, conf, cls_conf, cls_pred = out[0][0]
        bArea = (y2 - y1) * (x2 - x1)
        
        if bArea > 0.5:
            
            if cls_pred.item() in ban_list:
                return -1
            cls_high.append(int(cls_pred.item()) + 20)
            print(cls_high)
            if len(cls_high) == 10:
                ids = Counter(cls_high).most_common(n=1)[0][0]
                cls_high.clear()
                if (cls_before[0] != ids):
                    cls_before[0] = ids
                    return ids
                return -1
            
    return -1

def non_max_suppression(pred, conf_thres=0.9, nms_thres=0.6):

    pred[..., :4] = xywh2xyxy(pred[..., :4])
    output = [None for _ in range(len(pred))]
    
    for image_i, image_pred in enumerate(pred):
       
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
     
        if not image_pred.size(0):
            continue

        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
       
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
   
        keep_boxes = []
    
        while detections.size(0):
            
            large_overlap = calc_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def calc_iou(box1, box2, x1y1x2y2=True):

    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[0, 0] - box1[0, 2] / 2, box1[0, 0] + box1[0, 2] / 2
        b1_y1, b1_y2 = box1[0, 1] - box1[0, 3] / 2, box1[0, 1] + box1[0, 3] / 2
        b2_x1, b2_x2 = box2[0, 0] - box2[0, 2] / 2, box2[0, 0] + box2[0, 2] / 2
        b2_y1, b2_y2 = box2[0, 1] - box2[0, 3] / 2, box2[0, 1] + box2[0, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0, 0], box1[0, 1], box1[0, 2], box1[0, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0, 0], box2[0, 1], box2[0, 2], box2[0, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def xywh2xyxy(x):
    
    y = x.new(x.shape)
    
    y[..., 0] = x[..., 0] - x[..., 2] / 2 #x 
    y[..., 1] = x[..., 1] - x[..., 3] / 2 #y
    y[..., 2] = x[..., 0] + x[..., 2] / 2 #w
    y[..., 3] = x[..., 1] + x[..., 3] / 2 #h
    
    return y
