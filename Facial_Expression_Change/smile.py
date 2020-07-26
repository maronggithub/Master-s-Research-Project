import cv2
import dlib
import numpy as np
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)
from gaze_correction import GazeCorrector

PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

r = GazeCorrector(
            dlib_dat_path=PREDICTOR_PATH,
            model_dir="resources/models/gaze_correction/weights/warping_model/flx/12",
            screen_size_cm=(27.7, 17.9),
            screen_size_pt=(2560, 1600),
            app_window_rect=(2560 / 2, 1600/ 2, 640, 480),
            video_size=(640, 480),
            camera_pos_cm=(0, -9.4, 0),
            interpupillary_distance_cm=6.4,
            focal_length=400
        )

cap = cv2.VideoCapture(1 )
# Set camera resolution. The max resolution is webcam dependent
# so change it to a resolution that is both supported by your camera
# and compatible with your monitor
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# fullscreenset
cv2.namedWindow('parts', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('parts', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
smileOrNormal=1
while True:
    # VideoCaptureから1フレーム読み込む
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
            vec_p = np.empty([25, 2], dtype=int)
            # for p in range(10):
            # define the p points
            vec_p[0][0] = shape.part(48).x
            vec_p[0][1] = shape.part(48).y
            vec_p[1][0] = shape.part(54).x
            vec_p[1][1] = shape.part(54).y
            vec_p[2][0] = (shape.part(13).x-shape.part(3).x)/4+shape.part(3).x
            vec_p[2][1] = shape.part(3).y
            vec_p[3][0] = shape.part(13).x-(shape.part(13).x-shape.part(3).x)/4
            vec_p[3][1] = shape.part(3).y
            vec_p[4][0] = (shape.part(14).x-shape.part(2).x)/8+shape.part(2).x
            vec_p[4][1] = shape.part(2).y
            vec_p[5][0] = shape.part(14).x-(shape.part(14).x-shape.part(2).x)/8
            vec_p[5][1] = shape.part(2).y
            vec_p[6][0] = shape.part(40).x
            vec_p[6][1] = shape.part(40).y
            vec_p[7][0] = shape.part(47).x
            vec_p[7][1] = shape.part(47).y
            vec_p[8][0] = shape.part(41).x
            vec_p[8][1] = shape.part(41).y
            vec_p[9][0] = shape.part(46).x
            vec_p[9][1] = shape.part(46).y
            # upper lip
            vec_p[10][0] = shape.part(49).x
            vec_p[10][1] = shape.part(49).y
            vec_p[11][0] = shape.part(53).x
            vec_p[11][1] = shape.part(53).y
            vec_p[12][0] = shape.part(50).x
            vec_p[12][1] = shape.part(50).y
            vec_p[13][0] = shape.part(52).x
            vec_p[13][1] = shape.part(52).y
            vec_p[14][0] = shape.part(51).x
            vec_p[14][1] = shape.part(51).y
            #lower lip
            vec_p[15][0] = shape.part(59).x
            vec_p[15][1] = shape.part(59).y
            vec_p[16][0] = shape.part(55).x
            vec_p[16][1] = shape.part(55).y
            vec_p[17][0] = shape.part(58).x
            vec_p[17][1] = shape.part(58).y
            vec_p[18][0] = shape.part(56).x
            vec_p[18][1] = shape.part(56).y
            vec_p[19][0] = shape.part(57).x
            vec_p[19][1] = shape.part(57).y

            vec_p[20][0] = (shape.part(15).x-shape.part(1).x)/3+shape.part(1).x
            vec_p[20][1] = shape.part(1).y
            vec_p[21][0] = shape.part(15).x-(shape.part(15).x-shape.part(1).x)/3
            vec_p[21][1] = shape.part(1).y
            vec_p[22][0] = shape.part(61).x
            vec_p[22][1] = shape.part(61).y
            vec_p[23][0] = shape.part(63).x
            vec_p[23][1] = shape.part(63).y


            # define the q points
            vec_q = np.empty([25, 2], dtype=int)
            # for q in range(10):
            vec_q[0][0] = shape.part(48).x-5
            vec_q[0][1] = shape.part(48).y-7
            vec_q[1][0] = shape.part(54).x+5
            vec_q[1][1] = shape.part(54).y-7
            vec_q[2][0] = (shape.part(13).x-shape.part(3).x)/4+shape.part(3).x
            vec_q[2][1] = (shape).part(3).y
            vec_q[3][0] = (shape.part(13).x-(shape.part(13).x-shape.part(3).x)/4)
            vec_q[3][1] = shape.part(3).y
            vec_q[4][0] = (shape.part(14).x-shape.part(2).x)/8+shape.part(2).x
            vec_q[4][1] = shape.part(2).y
            vec_q[5][0] = (shape.part(14).x-(shape.part(14).x-shape.part(2).x)/8)
            vec_q[5][1] = shape.part(2).y
            vec_q[6][0] = shape.part(40).x+2
            vec_q[6][1] = shape.part(40).y-10
            vec_q[7][0] = shape.part(47).x-2
            vec_q[7][1] = shape.part(47).y-10
            vec_q[8][0] = shape.part(41).x
            vec_q[8][1] = shape.part(41).y-10
            vec_q[9][0] = shape.part(46).x
            vec_q[9][1] = shape.part(46).y-10

            vec_q[10][0] = shape.part(49).x
            vec_q[10][1] = shape.part(49).y-4
            vec_q[11][0] = shape.part(53).x
            vec_q[11][1] = shape.part(53).y-4
            vec_q[12][0] = shape.part(50).x
            vec_q[12][1] = shape.part(50).y-2
            vec_q[13][0] = shape.part(52).x
            vec_q[13][1] = shape.part(52).y-2
            vec_q[14][0] = shape.part(51).x
            vec_q[14][1] = shape.part(51).y-1
            vec_q[15][0] = shape.part(59).x-2
            vec_q[15][1] = shape.part(59).y-2
            vec_q[16][0] = shape.part(55).x+2
            vec_q[16][1] = shape.part(55).y-2
            vec_q[17][0] = shape.part(58).x-2
            vec_q[17][1] = shape.part(58).y-2
            vec_q[18][0] = shape.part(56).x+2
            vec_q[18][1] = shape.part(56).y-2
            vec_q[19][0] = shape.part(57).x
            vec_q[19][1] = shape.part(57).y-2
            vec_q[20][0] = (shape.part(15).x-shape.part(1).x)/3+shape.part(1).x
            vec_q[20][1] = shape.part(1).y-5
            vec_q[21][0] = shape.part(15).x - (shape.part(15).x - shape.part(1).x) / 3
            vec_q[21][1] = shape.part(15).y - 5

            vec_q[22][0] = shape.part(61).x
            vec_q[22][1] = shape.part(61).y
            vec_q[23][0] = shape.part(63).x
            vec_q[23][1] = shape.part(63).y

            p = np.array([
                [vec_p[0][0], vec_p[0][1]], [vec_p[1][0],vec_p[1][1]], [vec_p[2][0], vec_p[2][1]], [vec_p[3][0],vec_p[3][1]],
                [vec_p[4][0],vec_p[4][1]], [vec_p[5][0], vec_p[5][1]], [vec_p[6][0],vec_p[6][1]],[vec_p[7][0],vec_p[7][1]],
                [vec_p[8][0],vec_p[8][1]], [vec_p[9][0],vec_p[9][1]], [vec_p[10][0],vec_p[10][1]], [vec_p[11][0],vec_p[11][1]],
                [vec_p[12][0], vec_p[12][1]],[vec_p[13][0], vec_p[13][1]], [vec_p[14][0], vec_p[14][1]],
                [vec_p[15][0], vec_p[15][1]],[vec_p[16][0], vec_p[16][1]],[vec_p[17][0], vec_p[17][1]],
                [vec_p[18][0], vec_p[18][1]], [vec_p[19][0], vec_p[19][1]],[vec_p[20][0], vec_p[20][1]],[vec_p[21][0], vec_p[21][1]]


            ])

            q = np.array([
                [vec_q[0][0], vec_q[0][1]], [vec_q[1][0], vec_q[1][1]], [vec_q[2][0], vec_q[2][1]], [vec_q[3][0], vec_q[3][1]],
                [vec_q[4][0], vec_q[4][1]], [vec_q[5][0], vec_q[5][1]], [vec_q[6][0], vec_q[6][1]], [vec_q[7][0], vec_q[7][1]],
                [vec_q[8][0], vec_q[8][1]], [vec_q[9][0], vec_q[9][1]], [vec_q[10][0], vec_q[10][1]], [vec_q[11][0], vec_q[11][1]],
                [vec_q[12][0], vec_q[12][1]], [vec_q[13][0], vec_q[13][1]], [vec_q[14][0], vec_q[14][1]],[vec_q[15][0], vec_q[15][1]],
                [vec_q[16][0], vec_q[16][1]], [vec_q[17][0], vec_q[17][1]], [vec_q[18][0], vec_q[18][1]],[vec_q[19][0], vec_q[19][1]],
                [vec_q[20][0], vec_q[20][1]],[vec_q[21][0], vec_q[21][1]]

            ])

            if (smileOrNormal == 0):
                after = mls_rigid_deformation_inv(frame, p, p)
            elif (smileOrNormal == 1):
                after = mls_rigid_deformation_inv(frame, p, q)

    except Exception as e:
        pass

    cv2.imshow('parts', after)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
       break
    if key & 0xFF == ord('0'):
       smileOrNormal = 0
    if key & 0xFF == ord('1'):
       smileOrNormal = 1

