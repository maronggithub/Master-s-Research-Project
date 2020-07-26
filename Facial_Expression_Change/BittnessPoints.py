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
     # vec = np.empty([68, 2], dtype=int)
    # coords = np.zeros((68, 2), dtype=int)
    #  define the p points
     # lower lip points
     p1_l_x = shape.part(48).x
     p1_l_y = shape.part(48).y
     p1_r_x = shape.part(54).x
     p1_r_y = shape.part(54).y
     p2_l_x = shape.part(59).x
     p2_l_y = shape.part(59).y
     p2_r_x = shape.part(55).x
     p2_r_y = shape.part(55).y
     p3_l_x = shape.part(58).x
     p3_l_y = shape.part(58).y
     p3_r_x = shape.part(56).x
     p3_r_y = shape.part(56).y
     p3_m_x = shape.part(57).x
     p3_m_y = shape.part(57).y
     # inner upper lip points
     p4_l_x = shape.part(60).x
     p4_l_y = shape.part(60).y
     p4_r_x = shape.part(64).x
     p4_r_y = shape.part(64).y
     p5_u_l_x = shape.part(61).x
     p5_u_l_y = shape.part(61).y
     p5_u_r_x = shape.part(63).x
     p5_u_r_y = shape.part(63).y
     # inner lower lip points
     p5_d_l_x = shape.part(67).x
     p5_d_l_y = shape.part(67).y
     p5_d_m_x = shape.part(66).x
     p5_d_m_y = shape.part(66).y
     p5_d_r_x = shape.part(65).x
     p5_d_r_y = shape.part(65).y
     # lower eyelid points
     p6_l_x = shape.part(41).x
     p6_l_y = shape.part(41).y
     p6_r_x = shape.part(46).x
     p6_r_y = shape.part(46).y
     p7_l_x = shape.part(40).x
     p7_l_y = shape.part(40).y
     p7_r_x = shape.part(47).x
     p7_r_y = shape.part(47).y
     # the front eyebrow points
     p8_l_x = shape.part(20).x
     p8_l_y = shape.part(20).y
     p8_r_x = shape.part(23).x
     p8_r_y = shape.part(23).y
     p9_l_x = shape.part(21).x
     p9_l_y = shape.part(21).y
     p9_r_x = shape.part(22).x
     p9_r_y = shape.part(22).y
     # jaw points
     p10_x = (shape.part(10).x - shape.part(6).x) / 2 + shape.part(6).x
     p10_y = shape.part(6).y
     # bouth side of the mouth points
     p11_l_x = (shape.part(12).x - shape.part(4).x) / 6 + shape.part(4).x
     p11_l_y = shape.part(4).y
     p11_r_x = shape.part(12).x - (shape.part(12).x - shape.part(4).x) / 6
     p11_r_y = shape.part(4).y

     # p12_l_x = shape.part(40).x
     # p12_l_y = shape.part(40).y
     # p12_r_x = shape.part(47).x
     # p12_r_y = shape.part(47).y
     # p13_l_x = shape.part(41).x
     # p13_l_y = shape.part(41).y
     # p13_r_x = shape.part(46).x
     # p13_r_y = shape.part(46).y

     # upper eyelid points
     p14_l_x = shape.part(38).x
     p14_l_y = shape.part(38).y
     p14_r_x = shape.part(43).x
     p14_r_y = shape.part(43).y
     p15_l_x = shape.part(37).x
     p15_l_y = shape.part(37).y
     p15_r_x = shape.part(44).x
     p15_r_y = shape.part(44).y
     # the back eyebrow points
     p16_l_x = shape.part(19).x
     p16_l_y = shape.part(19).y
     p16_r_x = shape.part(24).x
     p16_r_y = shape.part(24).y
     # upper lip points
     p17_l_x = shape.part(49).x
     p17_l_y = shape.part(49).y
     p17_r_x = shape.part(53).x
     p17_r_y = shape.part(53).y
     p18_l_x = shape.part(50).x
     p18_l_y = shape.part(50).y
     p18_r_x = shape.part(52).x
     p18_r_y = shape.part(52).y

     # define the q points
     # lower lip points
     q1_l_x = shape.part(48).x-10
     q1_l_y = shape.part(48).y+15
     q1_r_x = shape.part(54).x+10
     q1_r_y = shape.part(54).y+15
     q2_l_x = shape.part(59).x-2
     q2_l_y = shape.part(59).y-3
     q2_r_x = shape.part(55).x+2
     q2_r_y = shape.part(55).y-3
     q3_l_x = shape.part(58).x-2
     q3_l_y = shape.part(58).y-4
     q3_r_x = shape.part(56).x+2
     q3_r_y = shape.part(56).y-4
     q3_m_x = shape.part(57).x
     q3_m_y = shape.part(57).y-12
     # inner upper lip points
     q4_l_x = shape.part(60).x-3
     q4_l_y = shape.part(60).y+4
     q4_r_x = shape.part(64).x+3
     q4_r_y = shape.part(64).y+4
     q5_u_l_x = shape.part(61).x-2
     q5_u_l_y = shape.part(61).y+3
     q5_u_r_x = shape.part(63).x+2
     q5_u_r_y = shape.part(63).y+3
     # inner lower lip points
     q5_d_l_x = shape.part(67).x-1
     q5_d_l_y = shape.part(67).y+1
     q5_d_m_x = shape.part(66).x
     q5_d_m_y = shape.part(66).y
     q5_d_r_x = shape.part(65).x+1
     q5_d_r_y = shape.part(65).y+1
     # lower eyelid points
     q6_l_x = shape.part(41).x+2
     q6_l_y = shape.part(41).y-5
     q6_r_x = shape.part(46).x-2
     q6_r_y = shape.part(46).y-5
     q7_l_x = shape.part(40).x
     q7_l_y = shape.part(40).y-5
     q7_r_x = shape.part(47).x
     q7_r_y = shape.part(47).y-5
     # the front eyebrow points
     q8_l_x = shape.part(20).x+12
     q8_l_y = shape.part(20).y+8
     q8_r_x = shape.part(23).x-12
     q8_r_y = shape.part(23).y+8
     q9_l_x = shape.part(21).x+15
     q9_l_y = shape.part(21).y+10
     q9_r_x = shape.part(22).x-15
     q9_r_y = shape.part(22).y+10
     # jaw points
     q10_x = (shape.part(10).x - shape.part(6).x) / 2 + shape.part(6).x
     q10_y = shape.part(6).y - 15
     # bouth side of the mouth points
     q11_l_x = (shape.part(12).x - shape.part(4).x) / 6 + shape.part(4).x - 4
     q11_l_y = shape.part(4).y + 4
     q11_r_x = shape.part(12).x - (shape.part(12).x - shape.part(4).x) / 6 + 4
     q11_r_y = shape.part(4).y + 4

     # q12_l_x = shape.part(40).x + 2
     # q12_l_y = shape.part(40).y - 12
     # q12_r_x = shape.part(47).x - 2
     # q12_r_y = shape.part(47).y - 12
     # q13_l_x = shape.part(41).x
     # q13_l_y = shape.part(41).y - 14
     # q13_r_x = shape.part(46).x
     # q13_r_y = shape.part(46).y - 14

     # upper eyelid points
     q14_l_x = shape.part(38).x+2
     q14_l_y = shape.part(38).y+1
     q14_r_x = shape.part(43).x-2
     q14_r_y = shape.part(43).y+1
     q15_l_x = shape.part(37).x
     q15_l_y = shape.part(37).y+1
     q15_r_x = shape.part(44).x
     q15_r_y = shape.part(44).y+1
     # the back eyebrow points
     q16_l_x = shape.part(19).x
     q16_l_y = shape.part(19).y+5
     q16_r_x = shape.part(24).x
     q16_r_y = shape.part(24).y+5
     # upper lip points
     q17_l_x = shape.part(49).x-1
     q17_l_y = shape.part(49).y+1
     q17_r_x = shape.part(53).x+1
     q17_r_y = shape.part(53).y+1
     q18_l_x = shape.part(50).x
     q18_l_y = shape.part(50).y+1
     q18_r_x = shape.part(52).x
     q18_r_y = shape.part(52).y+1


     # draw the p,q points

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
    cv2.circle(frame, (int(p5_u_l_x), int(p5_u_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q5_u_l_x), int(q5_u_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p5_u_r_x), int(p5_u_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q5_u_r_x), int(q5_u_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p5_d_l_x), int(p5_d_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q5_d_l_x), int(q5_d_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p5_d_m_x), int(p5_d_m_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q5_d_m_x), int(q5_d_m_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p5_d_r_x), int(p5_d_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q5_d_r_x), int(q5_d_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p6_l_x), int(p6_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q6_l_x), int(q6_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p6_r_x), int(p6_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q6_r_x), int(q6_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p7_l_x), int(p7_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q7_l_x), int(q7_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p7_r_x), int(p7_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q7_r_x), int(q7_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p8_l_x), int(p8_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q8_l_x), int(q8_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p8_r_x), int(p8_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q8_r_x), int(q8_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p9_l_x), int(p9_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q9_l_x), int(q9_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p9_r_x), int(p9_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q9_r_x), int(q9_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p10_x), int(p10_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q10_x), int(q10_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p11_l_x), int(p11_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q11_l_x), int(q11_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p11_r_x), int(p11_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q11_r_x), int(q11_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p12_l_x), int(p12_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q12_l_x), int(q12_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p12_r_x), int(p12_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q12_r_x), int(q12_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p13_l_x), int(p13_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q13_l_x), int(q13_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p13_r_x), int(p13_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q13_r_x), int(q13_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p14_l_x), int(p14_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q14_l_x), int(q14_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p14_r_x), int(p14_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q14_r_x), int(q14_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p15_l_x), int(p15_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q15_l_x), int(q15_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p15_r_x), int(p15_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q15_r_x), int(q15_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p16_l_x), int(p16_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q16_l_x), int(q16_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p16_r_x), int(p16_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q16_r_x), int(q16_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p17_l_x), int(p17_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q17_l_x), int(q17_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p17_r_x), int(p17_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q17_r_x), int(q17_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p18_l_x), int(p18_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q18_l_x), int(q18_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p18_r_x), int(p18_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q18_r_x), int(q18_r_y)), 3, (255, 0, 0), -1)
    cv2.imshow('parts', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()
