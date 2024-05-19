import cv2
import numpy as np
import concurrent.futures
from time import time

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

def frame_capture(cap, return_success=False):
    val_ret, img = cap.read()
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img if not return_success else (val_ret, img = cap.read(), img)

def ret_face(img, face_cascade,threshold_face, limit_no_detection):
    global counter_no_detection, face1, aoi

    if aoi is None:
        aoi = (0, 0, width, height)

    aoi_img = img[int(aoi[1]):int(aoi[1]) + int(aoi[3]),
                  int(aoi[0]):int(aoi[0]) + int(aoi[2])]
    
    gray_aoi_img = cv2.cvtColor(aoi_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_aoi_img, **detect_args)
    
    if len(faces) == 0:
        counter_no_detection += 1
        return None if counter_no_detection >= limit_no_detection else face1
    else:
        counter_no_detection = 0
        face = faces[0] if face1 is None else (faces[0] if not within_threshold(faces[0]) else face1)
        face1 = np.array([aoi[0] + face[0], aoi[1] + face[1], face[2], face[3]])
        return face1

def within_threshold(loc):
    global face1, cam_diag, threshold_face
    return np.all(np.abs(face1 - np.array(loc)) / cam_diag <= threshold_face)

def background_formatter(img, face_loc, rescale, mode, img_path_alt, kernel_size, mask_kernel_size, smooth_iters, iters_updation):
    global counter_update, masks, bg_model, fg_model, mask, sW, sH, kernel_back_blur, kernel_smooth

    counter_update += 1
    x, y, w, h = face_loc
    X, Y, W, H = [int(temp / rescale) for temp in (x, y, w, h)]

    if img.shape[:2] != (height, width):
        print('WARNING::BackgroundFormatter::ret_mask(): The shapes of image do not match (Camera Width, Camera Height)')
    
    small_img = cv2.resize(img, (sW, sH), interpolation=cv2.INTER_LINEAR)
    rectangle = (max(1, X - int(W)), max(1, Y - int(H)), min(int(3 * W), sW), small_img.shape[0] - (Y - int(H)))

    if (counter_update % iters_updation == 0):
        masks, bg_model, fg_model = cv2.grabCut(
            small_img, masks, rectangle, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT
        )
        mask = np.where((masks == 2) | (masks == 0), 0, 1).astype('uint8')
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.filter2D(mask, -1, kernel_smooth)
        mask = mask[:, :, np.newaxis]

    return mask

def apply_background(img, mask, mode, bg_img):
    if mode is None:
        return img
    elif mode == 'remove':
        bg_resized = cv2.resize(bg_img, (img.shape[1], img.shape[0]))
        return img * mask + bg_resized * (1 - mask)
    elif mode == 'blur':
        alt_img = cv2.resize(img, (int(width / 4), int(height / 4)))
        alt_img = cv2.filter2D(alt_img, -1, kernel_back_blur)
        alt_img = cv2.resize(alt_img, (width, height))
        return img * mask + alt_img * (1 - mask)

if __name__ == '__main__':
    # Set the desired values directly here
    mode = "blur"  # Change this to the desired mode ('remove', 'blur', None)
    bg_path = "/home/jetson/project/EE551-Mini-Project/infinite_void.jpg"  # Change this to the desired background image path

    width, height = (640, 480)
    scale = 8
    iters_update = 10
    kernel_blur_size = 9
    mask_kernel_smooth_size = 15
    mask_iters_smooth = 5

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    threshold_face = 0.1
    detect_args = {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30), 'flags': cv2.CASCADE_SCALE_IMAGE}
    rescale = 8
    limit_no_detection = 0

    counter_no_detection = 0
    face1 = None
    aoi = None
    cam_diag = np.sqrt(width * 2 + height * 2)

    sW = int(width / rescale)
    sH = int(height / rescale)
    kernel_size = kernel_blur_size
    mask_kernel_size = mask_kernel_smooth_size
    smooth_iters = mask_iters_smooth
    kernel_back_blur = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    kernel_smooth = np.ones((mask_kernel_size, mask_kernel_size), np.float32) / (mask_kernel_size ** 2)
    counter_update = -1
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    masks = np.zeros((int(width / rescale), int(height / rescale)))
    mask = masks

    c = -1
    T = time()
    timer = 0
    timer_steps = 10

    while True:
        c += 1
        img = frame_capture(cap)
        loc = ret_face(img, cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'), threshold_face, limit_no_detection)
        
        if loc is not None:
            mask = background_formatter(img, loc, rescale, mode, bg_path, blur_kernel_size, mask_kernel_smooth_size,
                                         mask_iters_smooth, iters_update)
            img = apply_background(img, mask, mode, cv2.imread(bg_path))
            cv2.imshow('camera', img)
        else:
            cv2.imshow('camera', img)

        if cv2.waitKey(1) == 27 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        t = time()
        timer += t - T
        if c % timer_steps == 0:
            print("TIME: {} | FPS: {}".format(timer / timer_steps, timer))
            timer = 0
        T = t

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
