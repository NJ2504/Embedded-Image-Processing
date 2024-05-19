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
 
# Laoding the image and harrcascade face to detect face.
def detect_faces(gray):
    overlay_img = cv2.imread('satoru_gojo_filter.png')
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
    return face_cascade.detectMultiScale(gray, 1.1, 4)
def overlay_filter(overlay_img, frame, face_x, face_y, face_w, face_h):
    #Scaling factor to adjust the size of the overlay image
    scaling_factor = 2.0
    #Coordinates to capture region of interest
    shift_y = int(0.75 * face_h)
    width = int(face_w * scaling_factor)
    height = int(face_h * scaling_factor)
    x_coord = face_x - (width - face_w) // 2
    y_coord = face_y - shift_y
    #Resizing overlay image.
    overlay_resized = cv2.resize(overlay, (width, height))
    overlay_color = overlay_resized[:, :, :3]
    alpha_mask = overlay_resized[:, :, 3] / 255.0
    x1, y1, x2, y2 = max(x_coord, 0), max(y_coord, 0), min(x_coord + width, frame.shape[1]), min(y_coord + height, frame.shape[0])
    if x_coord < 0:
        overlay_color = overlay_color[:, -x_coord:]
        alpha_mask = alpha_mask[:, -x_coord:]
    if new_y < 0:
        overlay_color = overlay_color[-y_coord:, :]
        alpha_mask = alpha_mask[-y_coord:, :]
    alpha_mask = cv2.resize(alpha_mask, (x2 - x1, y2 - y1))
    overlay_color = cv2.resize(overlay_color, (x2 - x1, y2 - y1))
    region = frame[y1:y2, x1:x2]
    # Vectorized implementation of overlay filter
    region[:] = (region * (1 - alpha_mask[..., None]) + overlay_color * alpha_mask[..., None]).astype(region.dtype)
    frame[y1:y2, x1:x2] = region 
    return frame
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in parallel using concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        faces = executor.submit(detect_faces, gray).result()
    for (x, y, w, h) in faces:
        frame = overlay_filter(overlay_img, frame, x, y, w, h)
    cv2.imshow('Webcam - Face Filter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

