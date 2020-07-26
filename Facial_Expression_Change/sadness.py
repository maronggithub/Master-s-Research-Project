import cv2
import dlib
import os
import numpy as np
import time
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)
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
            # put on the book :camera_pos=-7.4, focal_length= 500
            camera_pos_cm=(0, -24.4, 0),
            interpupillary_distance_cm=6.4,
            focal_length=700
        )

# # 顔を変形する前15sで真ん黒画面を見ながらまつ
# cv2.namedWindow('sadness', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('sadness', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# img = cv2.imread('data/black.jpg',1)
# cv2.imshow('sadness', img)
# cv2.waitKey(5000)
#
# # ビーフ声をだす
# os.system('afplay data/beep-01a.wav')

cap = cv2.VideoCapture(1)
# Set camera resolution. The max resolution is webcam dependent
# so change it to a resolution that is both supported by your camera
# and compatible with your monitor
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# fullscreenset
cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
SadOrNormal=1
while True:
    # VideoCaptureから1フレーム読み込む
    # startTime = time.time()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 1)
    # Our operations on the frame come here
    # for each detected face

    try:

         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         after = r.correct(frame, gray, dets)
    except Exception as e:
         after = frame

    try:
        for d in dets:
            # Get the landmarks/parts for the face in box d.
            shape = predictor(frame, d)
            vec_p = np.empty([22, 2], dtype=int)
            # define p points
            # mouth points
            vec_p[0][0] = shape.part(48).x
            vec_p[0][1] = shape.part(48).y
            vec_p[1][0] = shape.part(54).x
            vec_p[1][1] = shape.part(54).y
            vec_p[2][0] = shape.part(59).x
            vec_p[2][1] = shape.part(59).y
            vec_p[3][0] = shape.part(55).x
            vec_p[3][1] = shape.part(55).y
            vec_p[4][0] = shape.part(58).x
            vec_p[4][1] = shape.part(58).y
            vec_p[5][0] = shape.part(56).x
            vec_p[5][1] = shape.part(56).y
            vec_p[6][0] = shape.part(57).x
            vec_p[6][1] = shape.part(57).y
            # eyebrow points
            vec_p[7][0] = shape.part(20).x
            vec_p[7][1] = shape.part(20).y
            vec_p[8][0] = shape.part(23).x
            vec_p[8][1] = shape.part(23).y
            vec_p[9][0] = shape.part(21).x
            vec_p[9][1]= shape.part(21).y
            vec_p[10][0] = shape.part(22).x
            vec_p[10][1] = shape.part(22).y
            vec_p[11][0] = (shape.part(11).x-shape.part(5).x)/2+shape.part(5).x
            vec_p[11][1] = shape.part(5).y
            # eye corner points
            # vec_p[12][0] = shape.part(36).x
            # vec_p[12][1] = shape.part(36).y
            # vec_p[13][0] = shape.part(45).x
            # vec_p[13][1] = shape.part(45).y
            # upper eyelid ponints
            vec_p[14][0] = shape.part(37).x
            vec_p[14][1] = shape.part(37).y
            vec_p[15][0] = shape.part(44).x
            vec_p[15][1] = shape.part(44).y
            vec_p[16][0] = shape.part(38).x
            vec_p[16][1] = shape.part(38).y
            vec_p[17][0] = shape.part(43).x
            vec_p[17][1] = shape.part(43).y
            # upper eyelid between the brow
            # vec_p[18][0] = shape.part(37).x
            # vec_p[18][1] = shape.part(19).y + (shape.part(37).y - shape.part(19).y) / 2
            # vec_p[19][0] = shape.part(24).x
            # vec_p[19][1] = shape.part(24).y + (shape.part(44).y - shape.part(24).y) / 2
            # vec_p[20][0] = shape.part(36).x
            # vec_p[20][1] = shape.part(18).y + (shape.part(36).y - shape.part(18).y) / 2
            # vec_p[21][0] = shape.part(25).x
            # vec_p[21][1] = shape.part(25).y + (shape.part(45).y - shape.part(25).y) / 2

            # define the q points
            vec_q = np.empty([22, 2], dtype=int)
            # mouth pointsq
            vec_q[0][0] = shape.part(48).x-1
            vec_q[0][1] = shape.part(48).y+2
            vec_q[1][0] = shape.part(54).x+1
            vec_q[1][1] = shape.part(54).y+2
            vec_q[2][0] = shape.part(59).x
            vec_q[2][1] = shape.part(59).y-2
            vec_q[3][0] = shape.part(55).x
            vec_q[3][1] = shape.part(55).y-2
            vec_q[4][0] = shape.part(58).x
            vec_q[4][1] = shape.part(58).y-3
            vec_q[5][0] = shape.part(56).x
            vec_q[5][1] = shape.part(56).y-3
            vec_q[6][0] = shape.part(57).x
            vec_q[6][1] = shape.part(57).y-6
            # eyebrow points
            vec_q[7][0] = shape.part(20).x
            vec_q[7][1] = shape.part(20).y-8
            vec_q[8][0] = shape.part(23).x
            vec_q[8][1] = shape.part(23).y-8
            vec_q[9][0] = shape.part(21).x
            vec_q[9][1] = shape.part(21).y-10
            vec_q[10][0] = shape.part(22).x
            vec_q[10][1] = shape.part(22).y-10
            # lower lip to up, lower jaw point
            vec_q[11][0] = (shape.part(11).x - shape.part(5).x) / 2 + shape.part(5).x
            vec_q[11][1] = shape.part(5).y-4
            # eye corner points
            # vec_q[12][0] = shape.part(36).x
            # vec_q[12][1] = shape.part(36).y
            # vec_q[13][0] = shape.part(45).x
            # vec_q[13][1] = shape.part(45).y
            # upper eyelid ponints
            vec_q[14][0] = shape.part(37).x
            vec_q[14][1] = shape.part(37).y+3
            vec_q[15][0] = shape.part(44).x
            vec_q[15][1] = shape.part(44).y+3
            vec_q[16][0] = shape.part(38).x
            vec_q[16][1] = shape.part(38).y+2
            vec_q[17][0] = shape.part(43).x
            vec_q[17][1] = shape.part(43).y+2
            # upper eyelid between the brow
            # vec_q[18][0] = shape.part(37).x
            # vec_q[18][1] = shape.part(19).y + (shape.part(37).y - shape.part(19).y) / 2
            # vec_q[19][0] = shape.part(24).x
            # vec_q[19][1] = shape.part(44).y + (shape.part(44).y - shape.part(24).y) / 2
            # vec_q[20][0] = shape.part(36).x
            # vec_q[20][1] = shape.part(18).y + (shape.part(36).y - shape.part(18).y) / 2
            # vec_q[21][0] = shape.part(25).x
            # vec_q[21][1] = shape.part(25).y + (shape.part(45).y - shape.part(25).y) / 2


            p = np.array([
                [vec_p[0][0], vec_p[0][1]], [vec_p[1][0], vec_p[1][1]],[vec_p[2][0], vec_p[2][1]],[vec_p[3][0], vec_p[3][1]],
                [vec_p[4][0], vec_p[4][1]], [vec_p[5][0], vec_p[5][1]],[vec_p[6][0], vec_p[6][1]],[vec_p[7][0], vec_p[7][1]],
                [vec_p[8][0], vec_p[8][1]], [vec_p[9][0], vec_p[9][1]], [vec_p[10][0], vec_p[10][1]],[vec_p[11][0], vec_p[11][1]],
                # [vec_p[12][0],vec_p[12][1]],[vec_p[13][0],vec_p[13][1]],
                [vec_p[14][0], vec_p[14][1]],
                [vec_p[15][0],vec_p[15][1]],[vec_p[16][0],vec_p[16][1]],[vec_p[17][0], vec_p[17][1]]
                # [vec_p[18][0],vec_p[18][1]],[vec_p[19][0],vec_p[19][1]],
                # [vec_p[20][0],vec_p[20][1]],[vec_p[21][0],vec_p[21][1]]

            ])
            q = np.array([
                [vec_q[0][0], vec_q[0][1]], [vec_q[1][0], vec_q[1][1]],[vec_q[2][0], vec_q[2][1]],[vec_q[3][0], vec_q[3][1]],
                [vec_q[4][0], vec_q[4][1]], [vec_q[5][0], vec_q[5][1]],[vec_q[6][0], vec_q[6][1]],[vec_q[7][0], vec_q[7][1]],
                [vec_q[8][0], vec_q[8][1]], [vec_q[9][0], vec_q[9][1]],[vec_q[10][0], vec_q[10][1]],[vec_q[11][0], vec_q[11][1]],
                # [vec_q[12][0],vec_q[12][1]],[vec_q[13][0], vec_q[13][1]],
                [vec_q[14][0], vec_q[14][1]],
                [vec_q[15][0],vec_q[15][1]],[vec_q[16][0], vec_q[16][1]],[vec_q[17][0], vec_q[17][1]]
                # [vec_q[18][0],vec_q[18][1]],[vec_q[19][0], vec_q[19][1]],
                # [vec_q[20][0],vec_q[20][1]],[vec_q[21][0], vec_q[21][1]]
            ])

            if (SadOrNormal == 0):
                after = mls_rigid_deformation_inv(frame, p, p)
            elif (SadOrNormal == 1):
                after = mls_rigid_deformation_inv(frame, p, q)

    except Exception as e:
        print("error: ", e)

    cv2.imshow('1', after)
    key = cv2.waitKey(1)
    # print(f"Time: {time.time() - startTime} sec")
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('0'):
        SadOrNormal = 0
    if key & 0xFF == ord('1'):
        SadOrNormal = 1

    # # 1min後刺激を止まる
    # cv2.imshow('sadness', after)
    # cv2.waitKey(60000)
    # os.system('afplay data/beep-05.wav')
    # cv2.destroyWindow()

