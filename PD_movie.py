# adapt to movie based on PD_2016.py

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

def process_image(img):
    # initialize pedestrian detector
    hog = cv2.HOGDescriptor() #derive HOG features
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) #setSVMDetector
    # start pedestrian detection
    found, w = hog.detectMultiScale(img, winStride = (8,8), padding = (8, 8), scale = 1.15, finalThreshold = 1)
    draw_detections(img, found) # draw rectangles
    result = img
    return result

test_output = 'test_output.mp4'
clip1 = VideoFileClip('../Dataset/YouTube/【ドラレコ】横浜市営バスの恐怖2.mp4')
test_clip = clip1.fl_image(process_image)
test_clip.write_videofile(test_output, audio=False)
