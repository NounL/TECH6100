import numpy as np
import cv2
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    # resizing for faster detection
#     frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    detections = []
    detections = hog.detectMultiScale(gray, winStride=(8,8) )
    
#     print(detections)
    # create array "boxes" containing x , y position and width and height from x and y
#     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in detections[0]])

    
    for i in range(len(detections[0])):
        (x, y, w, h) = detections[0][i]
        (xA, yA, xB, yB) = (x, y, x+w, y+h)
        weight = detections[1][i]
        
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 255), 1)
        cv2.putText(frame, str(weight), (xA, yA),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)