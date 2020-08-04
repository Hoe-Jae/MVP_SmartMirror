import numpy as np
import torch, sys, os
import socket

from src.camera import Camera
from src.depth_camera import DepthCamera
from src.config import args
from src.step_motor import StepMotor
from src.models import YOLOv3
from src.utils import *

if __name__ == '__main__':
    
    print('[ Info ] MVP Project Initialize Start')
    rCap = Camera(args.rWidth, args.rHeight)
    if not rCap.isOpened():
        sys.exit('[ Error ] On Board Camera Init Failed')
        
    dCap = DepthCamera(args.dWidth, args.dHeight)
    if not dCap.isOpened():
        sys.exit('[ Error ] Depth Camera Init Failed')
        
    motor = StepMotor(args.mTop, args.mMid, args.mBot)
    
    print('[ Info ] YOLOv3 Model Load Start')
    model = YOLOv3(args.numClasses).cuda()
    res = model.load_state_dict(torch.load('./res/YOLOV3_200.pth'))
    if len(res.missing_keys) != 0 and len(res.unexpected_keys) != 0:
        sys.exit('[ Error ] YOLOv3 Model Load Failed')
    
    print('[ Info ] TCP Connect start')
    cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        cs.connect(('localhost', args.tcpPort))
    except:
        sys.exit('[ Error ] TCP Connect Failed')
        
    print('[ Info ] Background Extraction Start')
    BACKGND_MAPS = []
    BACKGND_DEPTH = np.zeros((args.dHeight, args.dWidth), np.uint16)
    while len(BACKGND_MAPS) != args.dBackCnt:
        
        dmap, d4d = dCap.read()
        BACKGND_MAPS.append(dmap)
    
    BACKGND_DEPTH = np.min(np.array(BACKGND_MAPS, np.uint16), axis=0)
    BACKGND_DEPTH[BACKGND_DEPTH <= 10] = args.dBackDef
    print('[ Info ] Background Extraction Finished')
    
    heightArr = []
    heightAvg = [0]
    heightOn = True
    cls_high = []
    cls_before = [-1]
    
    motor.move(2)
    while True:
        
        #motor.printState()
        dmap, _ = dCap.read()
        
        dmap[dmap >= BACKGND_DEPTH] = 2**12-1
        dmap[dmap >= 1500] = 2**12-1
        
        cDepth = dmap[args.dHeight //2, args.dWidth // 2]
        cDistance = np.cos(0.54) * cDepth
        
        cpDepth = np.min(dmap[dmap > 0].flatten())
        
        if cpDepth < 1350 and heightOn == True:
            
            height = 2160 - np.sqrt(cpDepth ** 2 - cDistance**2)

            
            if not np.isnan(height):
                
                heightArr.append(height)
                
                if len(heightArr) == 31:
                    arr = np.sort(heightArr)[3:10]
                    #print(arr)
                    heightAvg[0] = sum(arr)/len(arr)
                    
                    #heightAvg[0] = np.sort(heightArr)[15]
                    heightOn = False
                    print(heightAvg[0])
                    
                    if heightAvg[0] > 1650:
                        motor.move(0) 
                    elif 1500 > heightAvg[0]:
                        motor.move(2)
                    else:
                        motor.move(1)
          
        elif cpDepth > 1350 and heightOn == False:
            
            heightOn = True
            heightArr = []
            heightAvg[0] = 0
            arr = []
            
            motor.move(2)
            
            cs.send(('Home').encode('utf-8'))

        elif cpDepth < 1350 and heightOn == False:
            
            ret, img = rCap.read()
            
            if ret:
                
                yImg = cv2.resize(img, (416, 416))
                _, out = model(torch.from_numpy(yImg.transpose([2, 0, 1])).float().unsqueeze(0).cuda())
                out = non_max_suppression(out)
                res = detect(out, cls_high, cls_before)
                if res != -1:
                    cs.send(('DATA'+str(int(res))).encode('utf-8'))
                
                #cv2.imshow('RGB', yImg)
                #key = cv2.waitKey(1)
                #if key == 27:
                #    break
    
        d4d = np.uint8(dmap.astype(float) * 255 / 2**12-1)
        d4d = cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
        d4d = 255 - d4d

        if cpDepth < 1350:
            cpDepth_c, cpDepth_r = np.where(cpDepth == dmap)
       
            d4d[cpDepth_c, cpDepth_r, 0] = 0
            d4d[cpDepth_c, cpDepth_r, 1] = 0
            d4d[cpDepth_c, cpDepth_r, 2] = 255

        cv2.imshow("Depth", d4d) 
        key = cv2.waitKey(1)
        if key == 27:
            break
          
    rCap.release()
    dCap.release()
    motor.release()
    cs.close()
    
    print('[ Info ] MVP Project Finalized')

