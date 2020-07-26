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

    # mouth points
    p1_l_x=shape.part(48).x
    p1_l_y=shape.part(48).y
    p1_r_x=shape.part(54).x
    p1_r_y=shape.part(54).y
    p2_l_x=shape.part(59).x
    p2_l_y=shape.part(59).y
    p2_r_x=shape.part(55).x
    p2_r_y=shape.part(55).y
    p3_l_x=shape.part(58).x
    p3_l_y=shape.part(58).y
    p3_r_x=shape.part(56).x
    p3_r_y=shape.part(56).y
    p3_m_x=shape.part(57).x
    p3_m_y=shape.part(57).y
    # eyebrow points
    p4_l_x=shape.part(20).x
    p4_l_y=shape.part(20).y
    p4_r_x =shape.part(23).x
    p4_r_y =shape.part(23).y
    p5_l_x=shape.part(21).x
    p5_l_y=shape.part(21).y
    p5_r_x=shape.part(22).x
    p5_r_y=shape.part(22).y
    p6_x=(shape.part(11).x-shape.part(5).x)/2+shape.part(5).x
    p6_y=shape.part(5).y
    # eye corner points
    # p7_l_x=shape.part(36).x
    # p7_l_y=shape.part(36).y
    # p7_r_x=shape.part(45).x
    # p7_r_y=shape.part(45).y
    # upper eyelid ponints
    p8_l_x = shape.part(37).x
    p8_l_y = shape.part(37).y
    p8_r_x = shape.part(44).x
    p8_r_y = shape.part(44).y
    p9_l_x = shape.part(38).x
    p9_l_y = shape.part(38).y
    p9_r_x = shape.part(43).x
    p9_r_y = shape.part(43).y
    # upper eyelid between the brow
    # p10_l_x = shape.part(37).x
    # p10_l_y = shape.part(24).y+(shape.part(44).y-shape.part(24).y)/2
    # p10_r_x = shape.part(24).x
    # p10_r_y = shape.part(19).y+(shape.part(37).y-shape.part(19).y)/2
    # p11_l_x = shape.part(36).x
    # p11_l_y = shape.part(18).y + (shape.part(36).y - shape.part(18).y) / 2
    # p11_r_x = shape.part(25).x
    # p11_r_y = shape.part(25).y + (shape.part(45).y - shape.part(25).y) / 2

    # define the q points
    # mouth points
    q1_l_x=shape.part(48).x-1
    q1_l_y=shape.part(48).y+2
    q1_r_x=shape.part(54).x+1
    q1_r_y=shape.part(54).y+2
    q2_l_x=shape.part(59).x
    q2_l_y=shape.part(59).y-2
    q2_r_x=shape.part(55).x
    q2_r_y=shape.part(55).y-2
    q3_l_x=shape.part(58).x
    q3_l_y=shape.part(58).y-3
    q3_r_x=shape.part(56).x
    q3_r_y=shape.part(56).y-3
    q3_m_x=shape.part(57).x
    q3_m_y=shape.part(57).y-6
    # eyebrow points
    q4_l_x=shape.part(20).x
    q4_l_y=shape.part(20).y-8
    q4_r_x =shape.part(23).x
    q4_r_y =shape.part(23).y-8
    q5_l_x=shape.part(21).x
    q5_l_y=shape.part(21).y-10
    q5_r_x=shape.part(22).x
    q5_r_y=shape.part(22).y-10
    # lower lip to up, lower jaw point
    q6_x = (shape.part(11).x - shape.part(5).x) / 2 + shape.part(5).x
    q6_y = shape.part(5).y-4
    # eye corner points
    # q7_l_x = shape.part(36).x
    # q7_l_y = shape.part(36).y
    # q7_r_x = shape.part(45).x
    # q7_r_y = shape.part(45).y
    # upper eyelid ponints
    q8_l_x = shape.part(37).x
    q8_l_y = shape.part(37).y+3
    q8_r_x = shape.part(44).x
    q8_r_y = shape.part(44).y+3
    q9_l_x = shape.part(38).x
    q9_l_y = shape.part(38).y+2
    q9_r_x = shape.part(43).x
    q9_r_y = shape.part(43).y+2
    # upper eyelid between the brow
    # q10_l_x = shape.part(37).x
    # q10_l_y = shape.part(19).y + (shape.part(37).y - shape.part(19).y) / 2
    # q10_r_x = shape.part(24).x
    # q10_r_y = shape.part(24).y + (shape.part(44).y - shape.part(24).y) / 2
    # q11_l_x = shape.part(36).x
    # q11_l_y = shape.part(18).y + (shape.part(36).y - shape.part(18).y) / 2
    # q11_r_x = shape.part(25).x
    # q11_r_y = shape.part(25).y + (shape.part(45).y - shape.part(25).y) / 2

    #draw the p,q points
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
    cv2.circle(frame, (int(p3_m_x), int(p3_m_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(q3_m_x), int(q3_m_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(p4_l_x), int(p4_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q4_l_x), int(q4_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p4_r_x), int(p4_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q4_r_x), int(q4_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p5_l_x), int(p5_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q5_l_x), int(q5_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p5_r_x), int(p5_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q5_r_x), int(q5_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p6_x), int(p6_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q6_x), int(q6_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p7_l_x), int(p7_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q7_l_x), int(q7_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p7_r_x), int(p7_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q7_r_x), int(q7_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p8_l_x), int(p8_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q8_l_x), int(q8_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p8_r_x), int(p8_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q8_r_x), int(q8_r_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p9_l_x), int(p9_l_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q9_l_x), int(q9_l_y)), 3, (255, 0, 0), -1)
    cv2.circle(frame, (int(p9_r_x), int(p9_r_y)), 3, (0, 255, 255), -1)
    cv2.circle(frame, (int(q9_r_x), int(q9_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p10_l_x), int(p10_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q10_l_x), int(q10_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p10_r_x), int(p10_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q10_r_x), int(q10_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p11_l_x), int(p11_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q11_l_x), int(q11_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p11_r_x), int(p11_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q11_r_x), int(q11_r_y)), 3, (255, 0, 0), -1)
    cv2.imshow('parts', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()
