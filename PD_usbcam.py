# adapt to usbcam based on PD_2016.py

import cv2
import numpy as np
import datetime

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def process_image(img):
    # initial setting
    height, width = img.shape[:2] # obtain size info
    debug_space = cv2.resize(np.zeros((1, 1, 3), np.uint8), (width, height)) # create debug space
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_space, 'size: '+str(height)+'x'+str(width),(0, height//10), font, 0.8, (0,255,0),1) # debug

    # initialize pedestrian detector
    hog = cv2.HOGDescriptor() #derive HOG features
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) #setSVMDetector

    # start pedestrian detection
    start = datetime.datetime.now()
    found, w = hog.detectMultiScale(img, winStride = (8,8), padding = (8, 8), scale = 1.15, finalThreshold = 1)
    #print("[INFO] detection took: {}ms".format(int((datetime.datetime.now() - start).total_seconds()*1000)))
    cv2.putText(debug_space, "[INFO] detection took: {}ms".format(int((datetime.datetime.now() - start).total_seconds()*1000))
    ,(0, height//5), font, 0.8, (0,255,0),1) # debug
    draw_detections(img, found) # draw rectangles

    result = np.concatenate((img, debug_space), axis=1) # concat result and debug space
    return result

# capture usb cam
cap = cv2.VideoCapture(0)

while True:
    # read 1 frame from VideoCapture
    ret, frame = cap.read()
    # show raw image
    #cv2.imshow('Raw Frame', frame)
    # main process
    appliedPedDetect = process_image(frame)
    cv2.imshow('Result', appliedPedDetect)
    # wait 1ms for key input & break if k=27(esc)
    k = cv2.waitKey(1)
    if k==27:
        break

# release capture & close window
cap.release()
cv2.destroyAllWindows()
