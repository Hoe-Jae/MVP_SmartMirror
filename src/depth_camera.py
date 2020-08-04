import numpy as np
import cv2

from primesense import openni2
from primesense import _openni2 as c_api

class DepthCamera(object):
    
    def __init__(self, width, height):
        
        print('[ Info ] Depth Camera Initialized Start')
        
        self.width = width
        self.height = height
        
        openni2.initialize("/usr/local/lib/ni2/")
        
    def isOpened(self):
        
        if not openni2.is_initialized():
            return False
            
        self.device = openni2.Device.open_any()
        self.getDepthStream()
        
        self.depth_stream.start()

        return True
        
    def getDepthStream(self):
        
        self.depth_stream = self.device.create_depth_stream()
        
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=self.width, resolutionY=self.height, fps=30))
        self.depth_stream.set_mirroring_enabled(False)
    
    def read(self):
        
        dmap = np.fromstring( self.depth_stream.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(self.height, self.width)
        d4d = np.uint8(dmap.astype(float) * 255 / 2**12-1)
        d4d = cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
        d4d = 255-d4d
        
        return dmap, d4d
     
    def __del__(self):
        
        self.release()
    
    def release(self):
        
        self.depth_stream.stop()
        openni2.unload()
        

