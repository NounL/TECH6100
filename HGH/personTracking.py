'''
adapted from:
https://realpython.com/face-recognition-with-python/
https://thedatafrog.com/en/articles/human-detection-video/
'''

import numpy as np
import cv2
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# initialize facial cascade classifier
cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# open webcam and start cv window
cv2.startWindowThread()
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    # resizing and turn grayscale for faster detection
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image and return bouding box and weights
    detections = []
    detections = hog.detectMultiScale(gray, winStride=(8,8))

    # detect faces in the image and return bounding box
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5,
        minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    # loop over fulll body detections and draw boxes with 
    for i in range(len(detections[0])):
        (x, y, w, h) = detections[0][i]
        (xA, yA, xB, yB) = (x, y, x+w, y+h)
        weight = detections[1][i]
        
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 255), 1)
        cv2.putText(frame, str(weight), (xA, yA),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 2)
    
    # draw bounding box around estimated body (a body is about 7 heads tall and 1.5 heads wide)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,
                      (int(x-w/2),int(y-h/2)),
                      (int(x+w*1.5), int(y+h*7)),
                      (255,255,0), 1)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)