
# adapted from:
# https://realpython.com/face-recognition-with-python/
# https://thedatafrog.com/en/articles/human-detection-video/
# https://www.tutorialspoint.com/how-to-mask-an-image-in-opencv-python#


import cv2, sys
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
import time

LOCKED_ON = False

# setup steppers
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

motorPan = (2, 3, 4, 17)
motorTilt = (27, 22, 10, 9)

CCWStep = (0x01,0x02,0x04,0x08)
CWStep = (0x08,0x04,0x02,0x01)

def setup(motorPins):
    GPIO.setmode(GPIO.BCM)
    for pin in motorPins:
        GPIO.setup(pin,GPIO.OUT)

def moveOnePeriod(motorPins, direction,ms):
    for j in range(0,4,1):
        for i in range(0,4,1):
            if (direction == 1):
                GPIO.output(motorPins[i],((CCWStep[j] == 1<<i) and GPIO.HIGH or GPIO.LOW))
            else:
                GPIO.output(motorPins[i],((CWStep[j] == 1<<i) and GPIO.HIGH or GPIO.LOW))
        if(ms<3):
            ms = 3
        time.sleep(ms*0.001)

def moveSteps(motorPins, direction,ms,steps):
    for i in range(steps):
        moveOnePeriod(motorPins, direction,ms)

def motorStop(motorPins):
    for i in range(0,4,1):
        GPIO.output(motorPins[i],GPIO.LOW)

def loop():
    while True:
        moveSteps(motorPan, 1,3,75)
        time.sleep(0.5)
        moveSteps(motorPan, 0,3,75)
        time.sleep(0.5)
        moveSteps(motorTilt, 1,3,75)
        time.sleep(0.5)
        moveSteps(motorTilt, 0,3,75)
        time.sleep(0.5)

def destroy():
    GPIO.cleanup()

# setup buzzer
buzzer = 26
GPIO.setup(buzzer,GPIO.OUT)
buzzpwm = GPIO.PWM(buzzer,500)

def beep(x):
    for i in range(x):
        buzzpwm.start(50)
        sleep(0.1)
        buzzpwm.stop()
        sleep(0.1)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# initialize facial cascade classifier
cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# open webcam and start cv window
cv2.startWindowThread()
cap = cv2.VideoCapture(0)

setup(motorPan)
setup(motorTilt)

while(True):
    ret, frame = cap.read()

    # resizing and turn grayscale for faster detection
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # reset mask to blank output
    masked_img = np.zeros((480,640), np.uint8)
    
    # detect faces in the image and return bounding box
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5,
        minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    # beep if person not found
    if len(faces) == 0:
        LOCKED_ON = False
    
    # draw bounding box around estimated body (a body is about 7 heads tall and within 2 heads wide)
    for (x, y, w, h) in faces:

        # beep three times when person lock-on
        if LOCKED_ON == False:
            beep(3)
            LOCKED_ON = True
        
        (xA, yA, xB, yB) = (int(x-w),
                            int(y-h/2),
                            int(x+w*2),
                            int(y+h*7))
        # draw rectanngle
#         cv2.rectangle(frame, (xA,yA), (xB, yB), (255,255,0), 1)
#         cv2.rectangle(masked_img, (xA,yA), (xB, yB), (255,255,255), -1)
        
        # mask image using only detected areas
        masked_img = cv2.bitwise_and(gray, gray, mask = masked_img)
        
        personMidpoint = (int(xA+xB)/2,
                            int(yA+yB)/2)
        # tilt control
        # testing should be done from 5 to 6 feet away rom camera
        print(personMidpoint)
        if personMidpoint[0] < 280:
            print("panning left")
            moveSteps(motorTilt, 1, 3, 1)
        elif personMidpoint[0] > 340:
            moveSteps(motorTilt, 0, 3, 1)
            print("pan right")
        elif personMidpoint[1] < 400:
            print("tilting down")
            moveSteps(motorPan, 0, 3, 5)
        elif personMidpoint[1] > 500:
            moveSteps(motorPan, 1, 3, 5)
            print("tilting up")
        
    
    # detect people from masked image in and return bouding box and weights
    detections = []
    detections = hog.detectMultiScale(masked_img, winStride=(8,8))
    
    # loop over full body detections and draw boxes with 
    for i in range(len(detections[0])):
        (x, y, w, h) = detections[0][i]
        (xA, yA, xB, yB) = (x, y, x+w, y+h)
        weight = detections[1][i]
        
        # display the detected boxes in the colour picture
#         cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 255), 1)
#         cv2.putText(frame, str(weight), (xA, yA),
#                     cv2.FONT_HERSHEY_SIMPLEX, q0.2, (0,255,0), 2)
    
    
    # Display the resulting frame
#     cv2.imshow('frame',frame)
#     cv2.imshow("masked", masked_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
