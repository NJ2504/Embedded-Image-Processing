#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import concurrent.futures
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
face_haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_distort(face_region, max_distortion_factor=1.5):
    height, width, _ = face_region.shape
   # Creating a numpy mesh grid for x and y coordinates
    y_coord, x_coord = np.mgrid[0:height, 0:width]
    # Calculate distance of each pixel from the center pixel
    x_center, y_center = width // 2, height // 2
    # Calculate the Euclidean distance
    distance = np.sqrt((x_coord - x_center) **2 + (y_coord - y_center) **2)
    # Calculate the distortion factor based on the distance
    distortion_factor = 1 + (max_distortion_factor - 1) * (distance / distance.max())
    # Mapping back the distorted coordinates x_coord, y_coord, height and width to face region.
    x_distort = np.clip(x_center + (x_coord - x_center) * distortion_factor, 0, width - 1).astype(int)
    y_distort = np.clip(x_center + (y_coord - y_center) * distortion_factor, 0, height - 1).astype(int)
    # Creating distorted face
   distorted_face = face_region[y_distort, x_distort]
    return distorted_face
 cap = cv2.VideoCapture(0)
 while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    def process_face(face):
        (x, y, w, h) = face
        face_roi = frame[y:y + h, x:x + w]
        distorted_face = funny_face_distortion(face_roi, max_distortion_factor=1.5)
        frame[y:y + h, x:x + w] = distorted_face
 
    # Using concurrent futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_face, faces)
 
    cv2.imshow('Funny Face Distortion', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

