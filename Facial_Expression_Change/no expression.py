import cv2
import dlib
import os
import numpy as np
from gaze_correction import GazeCorrector

PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

frame_width = 640
frame_height = 480
resize_ration=0.5

r = GazeCorrector(
            dlib_dat_path=PREDICTOR_PATH,
            model_dir="resources/models/gaze_correction/weights/warping_model/flx/12",
            screen_size_cm=(43.5, 27.2),
            screen_size_pt=(1680, 1050),
            app_window_rect=(1680 / 2, 1050/ 2, 640, 480),
            video_size=(640, 480),
            # put on the book :camera_pos=-5.4, focal_length= 700
            # In the darkroom : camera_pose= -15.4,focal_length=700
            camera_pos_cm=(0, -15.4, 0),
            interpupillary_distance_cm=6.4,
            focal_length=700
        )

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# fullscreenset
cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 1)
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        after = r.correct(frame, gray, dets)
    except Exception as e:
        after = frame

    except Exception as e:
        print("error: ", e)

    cv2.imshow('1', after)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('0'):
        BittnessOrNormal = 0
    if key & 0xFF == ord('1'):
        BittnessOrNormal = 1