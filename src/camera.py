import queue as Q
import cv2
from   threading  import Thread

class Camera(object):
    
    def __init__(self, width, height):
        super(Camera, self).__init__()
        
        print('[ Info ] On Board Camera Initialized Start')
        
        gst_str = ('nvarguscamerasrc! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=0 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
        
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

        self.q = Q.Queue()
        t = Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except Q.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

    def isOpened(self):
        return self.cap.isOpened()
                
    def __del__(self):
        
        self.release()

    def release(self):

        self.cap.release()
