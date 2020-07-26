import cv2
import dlib
import numpy as np
import os
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)
from gaze_correction import GazeCorrector


PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

frame_width = 640
frame_height = 480

r = GazeCorrector(
            dlib_dat_path=PREDICTOR_PATH,
            model_dir="resources/models/gaze_correction/weights/warping_model/flx/12",
            screen_size_cm=(43.5, 27.2),
            screen_size_pt=(1680, 1050),
            app_window_rect=(1680 / 2, 1050/ 2, 640, 480),
            video_size=(640, 480),
            # put on the book :camera_pos=-5.4, focal_length= 700
            # In the darkroom : camera_pose= -15.4,focal_length=700
            camera_pos_cm=(0, -20.4, 0),
            interpupillary_distance_cm=6.4,
            focal_length=700
        )

# # 顔を変形する前15sで真ん黒画面を見ながらまつ
# cv2.namedWindow('bittness', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('bittness', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# img = cv2.imread('data/black.jpg',1)
# cv2.imshow('bittness', img)
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
BittnessOrNormal=1

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 1)
    # Our operations on the frame come here
    # for each detected face

    # try:
    #
    #      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #      after = r.correct(frame, gray, dets)
    # except Exception as e:
    #      after = frame

    try:
         for d in dets:
              # Get the landmarks/parts for the face in box d.
              shape = predictor(frame, d)
              vec_p = np.empty([40, 2], dtype=int)
              # define p points
              # lower lip points
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
              # inner upper lip points
              vec_p[7][0] = shape.part(60).x
              vec_p[7][1] = shape.part(60).y
              vec_p[8][0] = shape.part(64).x
              vec_p[8][1] = shape.part(64).y
              vec_p[9][0] = shape.part(61).x
              vec_p[9][1]= shape.part(61).y
              vec_p[10][0] = shape.part(63).x
              vec_p[10][1] = shape.part(63).y
              # inner lower lip points
              vec_p[11][0] = shape.part(67).x
              vec_p[11][1] = shape.part(67).y
              vec_p[12][0] = shape.part(66).x
              vec_p[12][1] = shape.part(66).y
              vec_p[13][0] = shape.part(65).x
              vec_p[13][1] = shape.part(65).y
              # lower eyelid points
              vec_p[14][0] = shape.part(41).x
              vec_p[14][1] = shape.part(41).y
              vec_p[15][0] = shape.part(46).x
              vec_p[15][1] = shape.part(46).y
              vec_p[16][0] = shape.part(40).x
              vec_p[16][1] = shape.part(40).y
              vec_p[17][0] = shape.part(47).x
              vec_p[17][1] = shape.part(47).y
              # the front eyebrow points
              vec_p[18][0] = shape.part(20).x
              vec_p[18][1] = shape.part(20).y
              vec_p[19][0] = shape.part(23).x
              vec_p[19][1] = shape.part(23).y
              vec_p[20][0] = shape.part(21).x
              vec_p[20][1] = shape.part(21).y
              vec_p[21][0] = shape.part(22).x
              vec_p[21][1] = shape.part(22).y
              # jaw points
              vec_p[22][0] = (shape.part(10).x - shape.part(6).x) / 2 + shape.part(6).x
              vec_p[22][1] = shape.part(6).y
              # bouth side of the mouth points
              vec_p[23][0] = (shape.part(12).x - shape.part(4).x) / 6 + shape.part(4).x
              vec_p[23][1] = shape.part(4).y
              vec_p[24][0] = shape.part(12).x - (shape.part(12).x - shape.part(4).x) / 6
              vec_p[24][1] = shape.part(4).y

              # vec_p[25][0] = shape.part(40).x
              # vec_p[25][1] = shape.part(40).y
              # vec_p[26][0] = shape.part(47).x
              # vec_p[26][1] = shape.part(47).y
              # vec_p[27][0] = shape.part(41).x
              # vec_p[27][1] = shape.part(41).y
              # vec_p[28][0] = shape.part(46).x
              # vec_p[28][1] = shape.part(46).y

              # upper eyelid points
              vec_p[29][0] = shape.part(38).x
              vec_p[29][1] = shape.part(38).y
              vec_p[30][0] = shape.part(43).x
              vec_p[30][1] = shape.part(43).y
              vec_p[31][0] = shape.part(37).x
              vec_p[31][1] = shape.part(37).y
              vec_p[32][0] = shape.part(44).x
              vec_p[32][1] = shape.part(44).y
              # the back eyebrow points
              vec_p[33][0] = shape.part(19).x
              vec_p[33][1] = shape.part(19).y
              vec_p[34][0] = shape.part(24).x
              vec_p[34][1] = shape.part(24).y
              # upper lip points
              vec_p[35][0] = shape.part(49).x
              vec_p[35][1] = shape.part(49).y
              vec_p[36][0] = shape.part(53).x
              vec_p[36][1] = shape.part(53).y
              vec_p[37][0] = shape.part(50).x
              vec_p[37][1] = shape.part(50).y
              vec_p[38][0] = shape.part(52).x
              vec_p[38][1] = shape.part(52).y

              # define the q points
              vec_q = np.empty([40, 2], dtype=int)
              # lower lip points
              vec_q[0][0] = shape.part(48).x-10
              vec_q[0][1] = shape.part(48).y+15
              vec_q[1][0] = shape.part(54).x+10
              vec_q[1][1] = shape.part(54).y+15
              vec_q[2][0] = shape.part(59).x-2
              vec_q[2][1] = shape.part(59).y-3
              vec_q[3][0] = shape.part(55).x+2
              vec_q[3][1] = shape.part(55).y-3
              vec_q[4][0] = shape.part(58).x-2
              vec_q[4][1] = shape.part(58).y-4
              vec_q[5][0] = shape.part(56).x+2
              vec_q[5][1] = shape.part(56).y-4
              vec_q[6][0] = shape.part(57).x
              vec_q[6][1] = shape.part(57).y-12
              # inner upper lip points
              vec_q[7][0] = shape.part(60).x-3
              vec_q[7][1] = shape.part(60).y+4
              vec_q[8][0] = shape.part(64).x+3
              vec_q[8][1] = shape.part(64).y+4
              vec_q[9][0] = shape.part(61).x-2
              vec_q[9][1] = shape.part(61).y+3
              vec_q[10][0] = shape.part(63).x+2
              vec_q[10][1] = shape.part(63).y+3
              # inner lower lip points
              vec_q[11][0] = shape.part(67).x-1
              vec_q[11][1] = shape.part(67).y+1
              vec_q[12][0] = shape.part(66).x
              vec_q[12][1] = shape.part(66).y
              vec_q[13][0] = shape.part(65).x+1
              vec_q[13][1] = shape.part(65).y+1
              # lower eyelid points
              vec_q[14][0] = shape.part(41).x+2
              vec_q[14][1] = shape.part(41).y-5
              vec_q[15][0] = shape.part(46).x-2
              vec_q[15][1] = shape.part(46).y-5
              vec_q[16][0] = shape.part(40).x
              vec_q[16][1] = shape.part(40).y-5
              vec_q[17][0] = shape.part(47).x
              vec_q[17][1] = shape.part(47).y-5
              # the front eyebrow points
              vec_q[18][0] = shape.part(20).x+12
              vec_q[18][1] = shape.part(20).y+8
              vec_q[19][0] = shape.part(23).x-12
              vec_q[19][1] = shape.part(23).y+8
              vec_q[20][0] = shape.part(21).x+15
              vec_q[20][1] = shape.part(21).y+10
              vec_q[21][0] = shape.part(22).x-15
              vec_q[21][1] = shape.part(22).y+10
              # jaw points
              vec_q[22][0] = (shape.part(10).x - shape.part(6).x) / 2 + shape.part(6).x
              vec_q[22][1] = shape.part(6).y-15
              # bouth side of the mouth points
              vec_q[23][0] = (shape.part(12).x - shape.part(4).x) / 6 + shape.part(4).x-4
              vec_q[23][1] = shape.part(4).y+4
              vec_q[24][0] = shape.part(12).x - (shape.part(12).x - shape.part(4).x) / 6+4
              vec_q[24][1] = shape.part(4).y+4

              # vec_q[25][0] = shape.part(40).x + 2
              # vec_q[25][1] = shape.part(40).y - 12
              # vec_q[26][0] = shape.part(47).x - 2
              # vec_q[26][1] = shape.part(47).y - 12
              # vec_q[27][0] = shape.part(41).x
              # vec_q[27][1] = shape.part(41).y - 14
              # vec_q[28][0] = shape.part(46).x
              # vec_q[28][1] = shape.part(46).y - 14
              # upper eyelid points
              vec_q[29][0] = shape.part(38).x+2
              vec_q[29][1] = shape.part(38).y+1
              vec_q[30][0] = shape.part(43).x-2
              vec_q[30][1] = shape.part(43).y+1
              vec_q[31][0] = shape.part(37).x
              vec_q[31][1] = shape.part(37).y+1
              vec_q[32][0] = shape.part(44).x
              vec_q[32][1] = shape.part(44).y+1
              # the back eyebrow points
              vec_q[33][0] = shape.part(19).x
              vec_q[33][1] = shape.part(19).y+5
              vec_q[34][0] = shape.part(24).x
              vec_q[34][1] = shape.part(24).y+5
              # upper lip points
              vec_q[35][0] = shape.part(49).x-1
              vec_q[35][1] = shape.part(49).y+1
              vec_q[36][0] = shape.part(53).x+1
              vec_q[36][1] = shape.part(53).y+1
              vec_q[37][0] = shape.part(50).x
              vec_q[37][1] = shape.part(50).y+1
              vec_q[38][0] = shape.part(52).x
              vec_q[38][1] = shape.part(52).y+1


              p = np.array([
                  [vec_p[0][0], vec_p[0][1]], [vec_p[1][0], vec_p[1][1]], [vec_p[2][0], vec_p[2][1]],[vec_p[3][0], vec_p[3][1]],
                  [vec_p[4][0], vec_p[4][1]], [vec_p[5][0], vec_p[5][1]], [vec_p[6][0], vec_p[6][1]],[vec_p[7][0], vec_p[7][1]],
                  [vec_p[8][0], vec_p[8][1]], [vec_p[9][0], vec_p[9][1]], [vec_p[10][0], vec_p[10][1]],[vec_p[11][0], vec_p[11][1]],
                  [vec_p[12][0], vec_p[12][1]], [vec_p[13][0], vec_p[13][1]], [vec_p[14][0], vec_p[14][1]],[vec_p[15][0], vec_p[15][1]],
                  [vec_p[16][0], vec_p[16][1]], [vec_p[17][0], vec_p[17][1]],
                  [vec_p[18][0], vec_p[18][1]],[vec_p[19][0], vec_p[19][1]],
                  [vec_p[20][0], vec_p[20][1]], [vec_p[21][0], vec_p[21][1]],
                  [vec_p[22][0], vec_p[22][1]],[vec_p[23][0], vec_p[23][1]],
                  [vec_p[24][0], vec_p[24][1]],
                  # [vec_p[25][0], vec_p[25][1]],[vec_p[26][0], vec_p[26][1]],
                  # [vec_p[27][0], vec_p[27][1]],[vec_p[28][0], vec_p[28][1]],
                  [vec_p[29][0], vec_p[29][1]], [vec_p[30][0], vec_p[30][1]],
                  [vec_p[31][0], vec_p[31][1]],[vec_p[32][0], vec_p[32][1]],
                  [vec_p[33][0], vec_p[33][1]], [vec_p[34][0], vec_p[34][1]],
                  [vec_p[35][0], vec_p[35][1]], [vec_p[36][0], vec_p[36][1]],
                  [vec_p[37][0], vec_p[37][1]], [vec_p[38][0], vec_p[38][1]]

              ])
              q = np.array([
                  [vec_q[0][0], vec_q[0][1]], [vec_q[1][0], vec_q[1][1]], [vec_q[2][0], vec_q[2][1]],[vec_q[3][0], vec_q[3][1]],
                  [vec_q[4][0], vec_q[4][1]], [vec_q[5][0], vec_q[5][1]], [vec_q[6][0], vec_q[6][1]],[vec_q[7][0], vec_q[7][1]],
                  [vec_q[8][0], vec_q[8][1]], [vec_q[9][0], vec_q[9][1]], [vec_q[10][0], vec_q[10][1]],[vec_q[11][0], vec_q[11][1]],
                  [vec_q[12][0], vec_q[12][1]], [vec_q[13][0], vec_q[13][1]], [vec_q[14][0], vec_q[14][1]],[vec_q[15][0], vec_q[15][1]],
                  [vec_q[16][0], vec_q[16][1]], [vec_q[17][0], vec_q[17][1]],
                  [vec_q[18][0], vec_q[18][1]],[vec_q[19][0], vec_q[19][1]],
                  [vec_q[20][0], vec_q[20][1]], [vec_q[21][0], vec_q[21][1]],
                  [vec_q[22][0], vec_q[22][1]],[vec_q[23][0], vec_q[23][1]],
                  [vec_q[24][0], vec_q[24][1]],
                  # [vec_q[25][0], vec_q[25][1]],[vec_q[26][0],vec_q[26][1]],
                  # [vec_q[27][0], vec_q[27][1]],[vec_q[28][0],vec_q[28][1]],
                  [vec_q[29][0], vec_q[29][1]],[vec_q[30][0], vec_q[30][1]],
                  [vec_q[31][0], vec_q[31][1]],[vec_q[32][0], vec_q[32][1]],
                  [vec_q[33][0], vec_q[33][1]],[vec_q[34][0], vec_q[34][1]],
                  [vec_q[35][0], vec_q[35][1]], [vec_q[36][0], vec_q[36][1]],
                  [vec_q[37][0], vec_q[37][1]], [vec_q[38][0], vec_q[38][1]]

              ])

              if (BittnessOrNormal == 0):
                   after = mls_rigid_deformation_inv(frame, p, p)
              elif (BittnessOrNormal == 1):
                   after = mls_rigid_deformation_inv(frame, p, q)

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

    # 1min後刺激を止まる
    # cv2.imshow('bittness', after)
    # cv2.waitKey(60000)
    # os.system('afplay data/beep-05.wav')
    # cv2.destroyWindow()


