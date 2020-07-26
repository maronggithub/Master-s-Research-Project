import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(1)
# Set camera resolution. The max resolution is webcam dependent
# so change it to a resolution that is both supported by your camera
# and compatible with your monitor
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# fullscreenset
cv2.namedWindow('parts', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('parts', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 1)
    # Our operations on the frame come here
    # for each detected face
    for d in dets:
        # Get the landmarks/parts for the face in box d.
     shape = predictor(frame, d)
       # define the p points
       # mouth corner points
     p1_l_x = shape.part(48).x
     p1_l_y = shape.part(48).y
     p1_r_x = shape.part(54).x
     p1_r_y = shape.part(54).y
     # cheek points
     p2_l_x = (shape).part(3).x + (shape.part(13).x - shape.part(3).x) / 5
     p2_l_y = (shape).part(3).y
     p2_r_x = shape.part(13).x - (shape.part(13).x - shape.part(3).x) / 5
     p2_r_y = shape.part(13).y
     p3_l_x = (shape.part(14).x - shape.part(2).x) / 4 + shape.part(2).x
     p3_l_y = shape.part(2).y
     p3_r_x = shape.part(14).x - (shape.part(14).x - shape.part(2).x) / 4
     p3_r_y = shape.part(14).y
     # lower eyelid points
     p4_l_x = shape.part(40).x
     p4_l_y = shape.part(40).y
     p4_r_x = shape.part(47).x
     p4_r_y = shape.part(47).y
     p5_l_x = shape.part(41).x
     p5_l_y = shape.part(41).y
     p5_r_x = shape.part(46).x
     p5_r_y = shape.part(46).y
     # upper lip points
     p6_l_x = shape.part(49).x
     p6_l_y = shape.part(49).y
     p6_r_x = shape.part(53).x
     p6_r_y = shape.part(53).y
     p7_l_x = shape.part(50).x
     p7_l_y = shape.part(50).y
     p7_r_x = shape.part(52).x
     p7_r_y = shape.part(52).y
     p8_x = shape.part(51).x
     p8_y = shape.part(51).y
     # lower lip points
     p9_l_x = shape.part(59).x
     p9_l_y = shape.part(59).y
     p9_r_x = shape.part(55).x
     p9_r_y = shape.part(55).y
     p10_l_x = shape.part(58).x
     p10_l_y = shape.part(58).y
     p10_r_x = shape.part(56).x
     p10_r_y = shape.part(56).y
     p11_x = shape.part(57).x
     p11_y = shape.part(57).y
     # lower eye inner corner points p
     # p12_l_x=(shape.part(15).x-shape.part(1).x)/3+shape.part(1).x   #upside the eye corner
     # p12_l_y=shape.part(1).y
     # p12_r_x=shape.part(15).x-(shape.part(15).x-shape.part(1).x)/3
     # p12_r_y=shape.part(1).y
     # inner upper lip points
     # p13_l_x=shape.part(39).x
     # p13_l_y=shape.part(39).y
     # p13_r_x=shape.part(42).x
     # p13_r_y=shape.part(42).y

     q1_l_x = shape.part(48).x - 5
     q1_l_y = shape.part(48).y - 7
     q1_r_x = shape.part(54).x + 5
     q1_r_y = shape.part(54).y - 7
     # cheek points
     q2_l_x = (shape.part(13).x - shape.part(3).x) / 3 + shape.part(3).x - 2
     q2_l_y = shape.part(3).y - 5
     q2_r_x = (shape.part(13).x - (shape.part(13).x - shape.part(3).x) / 3) + 2
     q2_r_y = shape.part(3).y - 5
     q3_l_x = (shape.part(14).x - shape.part(2).x) / 4 + shape.part(2).x
     q3_l_y = shape.part(2).y - 3
     q3_r_x = (shape.part(14).x - (shape.part(14).x - shape.part(2).x) / 4)
     q3_r_y = shape.part(2).y - 3
     # lower eyelid points
     q4_l_x = shape.part(40).x+1
     q4_l_y = shape.part(40).y - 1
     q4_r_x = shape.part(47).x + 1
     q4_r_y = shape.part(47).y - 1
     q5_l_x = shape.part(41).x
     q5_l_y = shape.part(41).y - 3
     q5_r_x = shape.part(46).x
     q5_r_y = shape.part(46).y - 3
     # upper lip points
     q6_l_x = shape.part(49).x
     q6_l_y = shape.part(49).y - 4
     q6_r_x = shape.part(53).x
     q6_r_y = shape.part(53).y - 4
     q7_l_x = shape.part(50).x
     q7_l_y = shape.part(50).y - 2
     q7_r_x = shape.part(52).x
     q7_r_y = shape.part(52).y - 2
     q8_x = shape.part(51).x
     q8_y = shape.part(51).y - 1
     # lower lip points
     q9_l_x = shape.part(59).x - 2
     q9_l_y = shape.part(59).y - 2
     q9_r_x = shape.part(55).x + 2
     q9_r_y = shape.part(55).y - 2
     q10_l_x = shape.part(58).x-2
     q10_l_y = shape.part(58).y - 2
     q10_r_x = shape.part(56).x+2
     q10_r_y = shape.part(56).y - 2
     q11_x = shape.part(57).x
     q11_y = shape.part(57).y - 2
     # lower eye inner corner points q
     # q12_l_x = (shape.part(15).x - shape.part(1).x) / 3 + shape.part(1).x
     # q12_l_y = shape.part(1).y
     # q12_r_x = shape.part(15).x - (shape.part(15).x - shape.part(1).x) / 3
     # q12_r_y = shape.part(1).y
     # inner upper lip points
     # q13_l_x = shape.part(39).x
     # q13_l_y = shape.part(39).y
     # q13_r_x = shape.part(42).x
     # q13_r_y = shape.part(42).y

     cv2.circle(frame, (int(p1_l_x), int(p1_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q1_l_x), int(q1_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p1_r_x), int(p1_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q1_r_x), int(q1_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p2_l_x), int(p2_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q2_l_x), int(q2_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p2_r_x), int(p2_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q2_r_x), int(q2_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p3_l_x), int(p3_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q3_l_x), int(q3_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p3_r_x), int(p3_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q3_r_x), int(q3_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p4_l_x), int(p4_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q4_l_x), int(q4_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p4_r_x), int(p4_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q4_r_x), int(q4_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p5_l_x), int(p5_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q5_l_x), int(q5_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p5_r_x), int(p5_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q5_r_x), int(q5_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p6_l_x), int(p6_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q6_l_x), int(q6_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p6_r_x), int(p6_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q6_r_x), int(q6_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p7_l_x), int(p7_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q7_l_x), int(q7_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p7_r_x), int(p7_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q7_r_x), int(q7_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p8_x), int(p8_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q8_x), int(q8_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p9_l_x), int(p9_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q9_l_x), int(q9_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p9_r_x), int(p9_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q9_r_x), int(q9_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p10_l_x), int(p10_l_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q10_l_x), int(q10_l_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p10_r_x), int(p10_r_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q10_r_x), int(q10_r_y)), 3, (255, 0, 0), -1)
     cv2.circle(frame, (int(p11_x), int(p11_y)), 3, (0, 255, 255), -1)
     cv2.circle(frame, (int(q11_x), int(q11_y)), 3, (255, 0, 0), -1)
     # cv2.circle(frame, (int(p12_l_x), int(p12_l_y)), 3, (0, 255, 255), -1)
     # cv2.circle(frame, (int(q12_l_x), int(q12_l_y)), 3, (255, 0, 0), -1)
     # cv2.circle(frame, (int(p12_r_x), int(p12_r_y)), 3, (0, 255, 255), -1)
     # cv2.circle(frame, (int(q12_r_x), int(q12_r_y)), 3, (255, 0, 0), -1)
     # cv2.circle(frame, (int(p13_l_x), int(p13_l_y)), 3, (0, 255, 255), -1)
     # cv2.circle(frame, (int(q13_l_x), int(q13_l_y)), 3, (255, 0, 0), -1)
     # cv2.circle(frame, (int(p13_r_x), int(p13_r_y)), 3, (0, 255, 255), -1)
     # cv2.circle(frame, (int(q13_r_x), int(q13_r_y)), 3, (255, 0, 0), -1)

     cv2.imshow('parts', frame)

     if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
