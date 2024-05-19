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
 
# Loading haarcascade eye filter
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Load the overlay eye image
overlay_image = cv2.imread('six_eyes.png', -1)  # -1 to load with alpha channel
# Check if the overlay image was loaded correctly
if overlay_image is None or overlay_image.shape[2] != 4:
    raise ValueError("Overlay image must be a PNG with an alpha channel.")
def overlay_filter(overlay, frame, eye_x, eye_y, eye_width, eye_height, scale=0.3):
    # Calculate height and width
    width = int(eye_width * scale)
    height = int(eye_height * scale)
   
    # Resize the overlay image according to the coordinates.
    overlay_resized = cv2.resize(overlay, (width, height))
   
    # Calculate the centers of the eyes
    x_center = eye_x + (eye_width - width) // 2
    y_center = eye_y + (eye_height - height) // 2
   
    # Separate the color and alpha channels of the overlay
    overlay_color = overlay_resized[:, :, :3]
    alpha_mask = overlay_resized[:, :, 3] / 255.0
   
    # Apply the overlay on the region of interest.
    region = frame[y_center : y_center + height, x_center : x_center + width]
   
    # Vectorized implementation
    region[:] = (region * (1 - alpha_mask[..., None]) + overlay_color * alpha_mask[..., None]).astype(region.dtype)
   
    frame[y_center : y_center + height, x_center : x_center + width] = region
    return frame
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
# Create a ThreadPoolExecutor with max_workers set to 2
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Converting the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    # Implementing Concurrent Futures
    futures = [executor.submit(overlay_filter, overlay_image, frame, ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes]
    concurrent.futures.wait(futures)
    cv2.imshow('Eye Filter Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
 

