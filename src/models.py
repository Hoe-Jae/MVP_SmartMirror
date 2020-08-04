import torch
import torch.nn as nn
import numpy as np

from src.basic_block import *

class DarkNet53(nn.Module):
    
    def __init__(self, num_classes):        
        super(DarkNet53, self).__init__()
        
        self.num_classes = num_classes

        ## Layer 1
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(32, 64, 3, 2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.1, inplace=True))
        self.layer2 = nn.Sequential(Residual(64, 32))
        
        ## Layer 2       
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.1, inplace=True))
        self.layer4 = nn.Sequential(Residual(128, 64),
                                   Residual(128, 64))
             
        ## Layer 3
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True))
        self.layer6 = nn.Sequential(Residual(256, 128),
                                   Residual(256, 128),
                                   Residual(256, 128),
                                   Residual(256, 128),
                                   Residual(256, 128),
                                   Residual(256, 128),
                                   Residual(256, 128),
                                   Residual(256, 128))
        
        ## Layer 4
        self.layer7 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1, inplace=True))
        self.layer8 = nn.Sequential(Residual(512, 256),
                                   Residual(512, 256),
                                   Residual(512, 256),
                                   Residual(512, 256),
                                   Residual(512, 256),
                                   Residual(512, 256),
                                   Residual(512, 256),
                                   Residual(512, 256))
        
        ## Layer 5
        self.layer9 = nn.Sequential(nn.Conv2d(512, 1024, 3, 2, padding=1),
                                   nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(0.1, inplace=True))
        
        self.layer10 = nn.Sequential(Residual(1024, 512),
                                    Residual(1024, 512),
                                    Residual(1024, 512),
                                    Residual(1024, 512))
        
    def load_darknet_weight(self):
        
        file_path = './checkpoints/darknet53.conv.74'

        with open(file_path, 'rb') as f:

            _ = np.fromfile(f, dtype=np.int32, count=3)
            _ = np.fromfile(f, dtype=np.int64, count=1)

            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for _, module in enumerate( self.children() ):

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Sigmoid):
                continue

            for i in range(len(module)):

                if isinstance(module[i], nn.Conv2d):
                    conv = module[i]

                elif isinstance(module[i], nn.BatchNorm2d):

                    bn = module[i]

                    nb = bn.bias.numel()
                    bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]).view_as(bn.bias))
                    ptr += nb

                    bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]).view_as(bn.weight))
                    ptr += nb

                    bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]).view_as(bn.running_mean))
                    ptr += nb

                    bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]).view_as(bn.running_var))
                    ptr += nb

                    nw = conv.weight.numel()
                    conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+nw]).view_as(conv.weight))
                    ptr += nw

                elif isinstance(module[i], Residual):

                    for m in module[i].children():

                        for j in range(len(m)):

                            if isinstance(m[j], nn.Conv2d):
                                conv = m[j]

                            elif isinstance(m[j], nn.BatchNorm2d):

                                bn = m[j]

                                nb = bn.bias.numel()
                                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]).view_as(bn.bias))
                                ptr += nb

                                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]).view_as(bn.weight))
                                ptr += nb

                                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]).view_as(bn.running_mean))
                                ptr += nb

                                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]).view_as(bn.running_var))
                                ptr += nb

                                nw = conv.weight.numel()
                                conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+nw]).view_as(conv.weight))
                                ptr += nw    
        
    def forward(self, x):
           
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            
        out_36 = self.layer6(self.layer5(x))
        out_61 = self.layer8(self.layer7(out_36))
        out_74 = self.layer10(self.layer9(out_61))
           
        return out_36, out_61, out_74
        
class YOLOv3(nn.Module):
    
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        
        self.num_classes = num_classes
        
        self.backbone = DarkNet53(num_classes)
        
        self.layer1 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(512, 1024, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(1024, 512, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(512, 1024, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(1024, 512, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1, inplace=True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(1024),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(1024, 60, 1, 1, 0, bias = False),
                                   nn.ReLU(inplace=True))
        # YOLO Layer 1
        self.yolo1 = YOLOLayer(32, self.num_classes)

        self.layer3 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Upsample(scale_factor=2))
        
        # route -1, 61 concat (layer3 output & output_61)
        
        self.layer4 = nn.Sequential(nn.Conv2d(512+256, 256, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(256, 512, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(512, 256, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(256, 512, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(512, 256, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True))

        self.layer5 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(512, 60, 1, 1, 0, bias = False),
                                   nn.ReLU(inplace=True))
        #YOLO Layer 2
        self.yolo2 = YOLOLayer(16, self.num_classes)
        
        self.layer6 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Upsample(scale_factor=2))
        
        # route -1, 36 concat (layer6 output & output_36)
        
        self.layer7 = nn.Sequential(nn.Conv2d(128+256, 128, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(128, 256, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(256, 128, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(128, 256, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(256, 128, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(128, 256, 3, 1, 1, bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(256, 60, 1, 1, 0, bias = False),
                                   nn.ReLU(inplace=True))
        
        self.yolo3 = YOLOLayer(8, self.num_classes)
               
    def forward(self, x, y=None):
        
        out_36, out_61, out_74 = self.backbone(x)
        # YOLO Layer 1
        r_1 = self.layer1(out_74)
        out = self.layer2(r_1)
        
        loss_1, out_1 = self.yolo1(out, y)
        
        # YOLO Layer 2
        out = self.layer3(r_1)
        r_2 = torch.cat([out, out_61], dim=1)
        r_3 = self.layer4(r_2)
        out = self.layer5(r_3)
        
        loss_2, out_2 = self.yolo2(out, y)
        
        # YOLO Layer 3
        out = self.layer6(r_3)
        out = torch.cat([out, out_36], dim=1)
        out = self.layer7(out)
        
        loss_3, out_3 = self.yolo3(out, y)
        
        out = torch.cat([out_1, out_2, out_3], dim=1)
        if y is None:
            return None, out
        
        loss = sum([loss_1, loss_2, loss_3])
        
        return loss, out
