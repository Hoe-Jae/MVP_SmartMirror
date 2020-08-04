import torch
import torch.nn as nn
import numpy as np

class Residual(nn.Module):
    def __init__(self, filter_in, filter_out):
        super(Residual, self).__init__()
        
        self.filter_in = filter_in
        self.filter_out = filter_out
        
        self.Conv1 = nn.Sequential(nn.Conv2d(self.filter_in, self.filter_out, 1, 1, padding=0),
                                  nn.BatchNorm2d(self.filter_out),
                                  nn.LeakyReLU(0.1, inplace=True))
        
        self.Conv2 = nn.Sequential(nn.Conv2d(self.filter_out, 2 * self.filter_out, 3, 1, padding=1),
                                  nn.BatchNorm2d(2 * self.filter_out),
                                  nn.LeakyReLU(0.1, inplace=True))
        
    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(out)
        out = x + out

        return out
    
class YOLOLayer(nn.Module):
    
    def __init__(self, scale, num_classes, ignore_thres=0.5):
        super(YOLOLayer, self).__init__()
        
        self.scale = scale
        self.ignore_thres = ignore_thres
        self.num_classes = num_classes
        
        self.L1Loss = nn.L1Loss(reduction='sum')
        self.BCELoss = nn.BCELoss(reduction='sum')
        
        self.softmax = nn.Softmax(dim=4)
        self.sigmoid = nn.Sigmoid()
        
        if self.scale == 32:
            self.anchors = [(318, 370), (276, 285), (155, 330)]
        elif self.scale == 16:
            self.anchors = [(201, 187), (135, 267), (84, 286)]
        elif self.scale == 8:
            self.anchors = [(96, 230), (79, 186), (81, 133)]
            
        self.grid_anchor = np.array([(w / self.scale, h / self.scale) for w, h in self.anchors])
        
    def calc_wh_iou(self, wh1, wh2):
        
        wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
        
        return inter_area / union_area
    
    def calc_iou(self, box1, box2, x1y1x2y2=True):

        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

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
        
    def build_targets(self, out_box, out_cls, target):
        
        # out_box : (b, g, g, 15)
        # out_cls : (b, g, g, 45)
        # target  : (b, 5)  x, y, w, h, cls
        
        BoolTensor = torch.cuda.BoolTensor if out_box.is_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if out_box.is_cuda else torch.FloatTensor
        
        nB = out_box.size(0)
        nG = out_box.size(2)
        
        out_box = out_box.view(nB, 3, nG, nG, 4)
        out_cls = out_cls.view(nB, 3, nG, nG, 15)
        
        nC = out_cls.size(-1)
        nA = out_box.size(1)
        
        # Output Tensors
        
        obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
        cls_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
        
        # Convert to position relative to box
        target_boxs = target[:, :4] * nG
        gxy = target_boxs[:, :2]
        gwh = target_boxs[:, 2:]
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        gcls = target[:,4]
        
        grid_anchor = FloatTensor(self.grid_anchor)
        ious = torch.stack([self.calc_wh_iou(anchor, gwh) for anchor in grid_anchor])
        best_iou, best_idx = ious.max(0)
        
        for b, (iou, idx, i, j, x, y, w, h, cls) in enumerate(zip(best_iou, best_idx, gi, gj, gx, gy, gw, gh, gcls)):
    
            obj_mask[b, idx, j, i] = 1
            noobj_mask[b, idx, j, i] = 0

            tx[b, idx, j, i] = x - i
            ty[b, idx, j, i] = y - j    
            tw[b, idx, j, i] = torch.log( w / grid_anchor[idx, 0] + 1e-16 )  
            th[b, idx, j, i] = torch.log( h / grid_anchor[idx, 1] + 1e-16 )

            tcls[b, idx, j, i, int(cls)] = 1

            cls_mask[b, idx, j, i] = (out_cls[b, idx, j, i].argmax(-1) == int(cls)).float()
            iou_scores[b, idx, j, i] = self.calc_iou(out_box[b, idx, j, i], target_boxs[b], x1y1x2y2=False)

        for b, (iou, i, j)in enumerate(zip(ious.t(), gi, gj)):

            noobj_mask[b, iou > self.ignore_thres, j, i] = 0

        tconf = obj_mask.float()
        
        return iou_scores, cls_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
        
    def forward(self, x, labels=None):
        
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        BoolTensor = torch.cuda.BoolTensor if x.is_cuda else torch.BoolTensor
        
        img_dim = 416
        nB = x.size(0)
        nG = x.size(2)
        
        pred = x.view(nB, 3, self.num_classes + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        
        x, y = self.sigmoid(pred[..., 0]), self.sigmoid(pred[..., 1])
        w, h = pred[..., 2], pred[..., 3]
        conf = self.sigmoid(pred[..., 4])
        cls = self.softmax(pred[..., 5:])
        
        pred_boxs = FloatTensor(pred[..., :4].shape)
        pred_boxs[..., 0] = x
        pred_boxs[..., 1] = y
        pred_boxs[..., 2] = w
        pred_boxs[..., 3] = h
        
        out = torch.cat([pred_boxs.view(nB, -1, 4) * self.scale, conf.view(nB, -1, 1), cls.view(nB, -1, self.num_classes)], dim=-1)
        
        if labels is None:
            
            return None, out
        
        iou_socore, cls_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(pred_boxs, cls, labels)
        
        loss_x = self.L1Loss(x[obj_mask], tx[obj_mask])
        loss_y = self.L1Loss(y[obj_mask], ty[obj_mask])
        loss_w = self.L1Loss(w[obj_mask], tw[obj_mask])
        loss_h = self.L1Loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.BCELoss(conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.BCELoss(conf[noobj_mask], tconf[noobj_mask])
        loss_conf = loss_conf_obj + 100 * loss_conf_noobj
        loss_cls = self.BCELoss(cls[obj_mask], tcls[obj_mask])
        
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls        
        
        return loss, out
