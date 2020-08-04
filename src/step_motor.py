import Jetson.GPIO as GPIO

class StepMotor(object):
    
    def __init__(self, TOP, MID, BOT):
        super(StepMotor, self).__init__()
        
        print('[ Info ] Step Motor Initialized Start')
        
        self.channels = [TOP, MID, BOT]
        self.state = [0, 0, 0]
        
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.channels, GPIO.OUT, initial=GPIO.LOW)
        
    def move(self, idx):
        
        if self.state[idx] == 1:
            self.state[idx] = 0
            GPIO.output(self.channels[idx], GPIO.LOW)
        else:
            self.state[idx] = 1
            GPIO.output(self.channels[idx], GPIO.HIGH)

    def printState(self):

        print(self.state)      
            
    def __del__(self):
        
        self.release()
    
    def release(self):
        
        GPIO.cleanup()
