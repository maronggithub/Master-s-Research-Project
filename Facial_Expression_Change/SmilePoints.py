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
    #  define the p points
    # mouth corner points
    p1_l_x=shape.part(48).x
    p1_l_y=shape.part(48).y
    p1_r_x=shape.part(54).x
    p1_r_y=shape.part(54).y
    # cheek points
    p2_l_x=(shape).part(3).x+(shape.part(13).x-shape.part(3).x)/5
    p2_l_y=(shape).part(3).y
    p2_r_x=shape.part(13).x-(shape.part(13).x-shape.part(3).x)/5
    p2_r_y=shape.part(3).y
    p3_l_x=(shape.part(14).x-shape.part(2).x)/6+shape.part(2).x
    p3_l_y=shape.part(2).y
    p3_r_x=shape.part(14).x-(shape.part(14).x-shape.part(2).x)/6
    p3_r_y=shape.part(2).y
    # lower eyelid points
    p4_l_x=shape.part(40).x
    p4_l_y=shape.part(40).y
    p4_r_x =shape.part(47).x
    p4_r_y =shape.part(47).y
    p5_l_x=shape.part(41).x
    p5_l_y=shape.part(41).y
    p5_r_x=shape.part(46).x
    p5_r_y=shape.part(46).y
    # upper lip points
    p6_l_x=shape.part(49).x
    p6_l_y=shape.part(49).y
    p6_r_x= shape.part(53).x
    p6_r_y=shape.part(53).y
    p7_l_x=shape.part(50).x
    p7_l_y=shape.part(50).y
    p7_r_x=shape.part(52).x
    p7_r_y=shape.part(52).y
    p8_x=shape.part(51).x
    p8_y=shape.part(51).y
    # lower eye inner corner points q
    # p12_l_x=(shape.part(15).x-shape.part(1).x)/3+shape.part(1).x   #upside the eye corner
    # p12_l_y=shape.part(1).y
    # p12_r_x=shape.part(15).x-(shape.part(15).x-shape.part(1).x)/3
    # p12_r_y=shape.part(1).y
    # inner upper lip points
    # p13_l_x=shape.part(61).x
    # p13_l_y=shape.part(61).y
    # p13_r_x=shape.part(63).x
    # p13_r_y=shape.part(63).y
    # eyebrow points
    # p14_l_x=shape.part(21).x
    # p14_l_y=shape.part(21).y
    # p14_r_x = shape.part(22).x
    # p14_r_y = shape.part(22).y
    # p15_l_x= shape.part(20).x
    # p15_l_y= shape.part(20).y
    # p15_r_x = shape.part(23).x
    # p15_r_y = shape.part(23).y
    # upper eyelipper
    # p16_l_x = shape.part(19).x
    # p16_l_y = shape.part(37).y-(shape.part(37).y-shape.part(19).y)/3
    # p16_r_x = shape.part(24).x
    # p16_r_y = shape.part(44).y - (shape.part(44).y - shape.part(24).y) / 3
    # upper eyelid points
    # p17_l_x = shape.part(38).x
    # p17_l_y = shape.part(38).y
    # p17_r_x = shape.part(43).x
    # p17_r_y = shape.part(43).y
    # p18_l_x = shape.part(37).x
    # p18_l_y = shape.part(37).y
    # p18_r_x = shape.part(44).x
    # p18_r_y = shape.part(44).y
    # p19_l_x = shape.part(38).x
    # p19_l_y = shape.part(38).y
    # p19_r_x = shape.part(43).x
    # p19_r_y = shape.part(43).y
    # p20_l_x = shape.part(37).x
    # p20_l_y = shape.part(37).y
    # p20_r_x = shape.part(44).x
    # p20_r_y = shape.part(44).y
    # back eyebrow points
    # p21_l_x = shape.part(19).x
    # p21_l_y = shape.part(19).y
    # p21_r_x = shape.part(24).x
    # p21_r_y = shape.part(24).y
    # p22_l_x = shape.part(18).x
    # p22_l_y = shape.part(18).y
    # p22_r_x = shape.part(25).x
    # p22_r_y = shape.part(25).y
    # p23_l_x = shape.part(17).x
    # p23_l_y = shape.part(17).y
    # p23_r_x = shape.part(26).x
    # p23_r_y = shape.part(26).y
    # below the corner of the eye2
    # p24_l_x = shape.part(47).x
    # p24_l_y = shape.part(28).y
    # p24_r_x = shape.part(40).x
    # p24_r_y = shape.part(28).y

    # define the q points
    # mouth corner points
    q1_l_x=shape.part(48).x-5
    q1_l_y=shape.part(48).y-8
    q1_r_x=shape.part(54).x+5
    q1_r_y=shape.part(54).y-8
    # cheek points
    q2_l_x=(shape.part(13).x-shape.part(3).x)/5+shape.part(3).x-5
    q2_l_y=shape.part(3).y-10
    q2_r_x=(shape.part(13).x-(shape.part(13).x-shape.part(3).x)/5)+5
    q2_r_y=shape.part(3).y-10
    q3_l_x=(shape.part(14).x-shape.part(2).x)/6+shape.part(2).x-10
    q3_l_y=shape.part(2).y-15
    q3_r_x=(shape.part(14).x-(shape.part(14).x-shape.part(2).x)/6)+10
    q3_r_y=shape.part(2).y-15
    # lower eyelid points
    q4_l_x=shape.part(40).x-5
    q4_l_y=shape.part(39).y-25
    q4_r_x =shape.part(47).x+5
    q4_r_y =shape.part(42).y-25
    q5_l_x=shape.part(41).x
    q5_l_y=shape.part(39).y-25
    q5_r_x=shape.part(46).x
    q5_r_y=shape.part(42).y-25
    # upper lip points
    q6_l_x = shape.part(49).x-6
    q6_l_y = shape.part(49).y-4
    q6_r_x = shape.part(53).x+6
    q6_r_y = shape.part(53).y-4
    q7_l_x = shape.part(50).x
    q7_l_y = shape.part(50).y-3
    q7_r_x = shape.part(52).x
    q7_r_y = shape.part(52).y-3
    q8_x = shape.part(51).x
    q8_y = shape.part(51).y-2
    # lower lip points
    q9_l_x = shape.part(59).x-2
    q9_l_y = shape.part(59).y-3
    q9_r_x = shape.part(55).x+2
    q9_r_y = shape.part(55).y-3
    q10_l_x = shape.part(58).x
    q10_l_y = shape.part(58).y-5
    q10_r_x = shape.part(56).x
    q10_r_y = shape.part(56).y-5
    q11_x = shape.part(57).x
    q11_y = shape.part(57).y-3
    # lower eye inner corner points q
    # q12_l_x = (shape.part(15).x - shape.part(1).x) / 3 + shape.part(1).x
    # q12_l_y = shape.part(1).y
    # q12_r_x = shape.part(15).x - (shape.part(15).x - shape.part(1).x) / 3
    # q12_r_y = shape.part(1).y
    # inner upper lip points
    # q13_l_x = shape.part(61).x-2
    # q13_l_y = shape.part(61).y-3
    # q13_r_x = shape.part(63).x+2
    # q13_r_y = shape.part(63).y-3
    # front eyebrow points
    # q14_l_x = shape.part(21).x
    # q14_l_y = shape.part(21).y+5
    # q14_r_x = shape.part(22).x
    # q14_r_y = shape.part(22).y+5
    # q15_l_x = shape.part(20).x
    # q15_l_y = shape.part(20).y+3
    # q15_r_x = shape.part(23).x
    # q15_r_y = shape.part(23).y+3
    # upper eyelipper
    # q16_l_x = shape.part(19).x
    # q16_l_y = shape.part(37).y - (shape.part(37).y - shape.part(19).y) / 3+4
    # q16_r_x = shape.part(24).x
    # q16_r_y = shape.part(44).y - (shape.part(44).y - shape.part(24).y) / 3+4
    # upper eyelid points
    # q17_l_x = shape.part(38).x
    # q17_l_y = shape.part(38).y+1
    # q17_r_x = shape.part(43).x
    # q17_r_y = shape.part(43).y+1
    # q18_l_x = shape.part(37).x
    # q18_l_y = shape.part(37).y+1
    # q18_r_x = shape.part(44).x
    # q18_r_y = shape.part(44).y+1
    # q19_l_x = shape.part(38).x
    # q19_l_y = shape.part(38).y+1
    # q19_r_x = shape.part(43).x
    # q19_r_y = shape.part(43).y+1
    # q20_l_x = shape.part(37).x
    # q20_l_y = shape.part(37).y+1
    # q20_r_x = shape.part(44).x
    # q20_r_y = shape.part(44).y+1
    # back eyebrow points
    # q21_l_x = shape.part(19).x
    # q21_l_y = shape.part(19).y + 3
    # q21_r_x = shape.part(24).x
    # q21_r_y = shape.part(24).y + 3
    # q22_l_x = shape.part(18).x
    # q22_l_y = shape.part(18).y + 1
    # q22_r_x = shape.part(25).x
    # q22_r_y = shape.part(25).y + 1
    # q23_l_x = shape.part(17).x
    # q23_l_y = shape.part(17).y + 1
    # q23_r_x = shape.part(26).x
    # q23_r_y = shape.part(26).y + 1
    # draw the p,q points
    # below the corner of the eye2
    # q24_l_x = shape.part(47).x
    # q24_l_y = shape.part(28).y
    # q24_r_x = shape.part(40).x
    # q24_r_y = shape.part(28).y

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
    # cv2.circle(frame, (int(p14_l_x), int(p14_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q14_l_x), int(q14_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p14_r_x), int(p14_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q14_r_x), int(q14_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p15_l_x), int(p15_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q15_l_x), int(q15_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p15_r_x), int(p15_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q15_r_x), int(q15_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p16_l_x), int(p16_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q16_l_x), int(q16_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p16_r_x), int(p16_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q16_r_x), int(q16_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p19_l_x), int(p19_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q19_l_x), int(q19_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p19_r_x), int(p19_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q19_r_x), int(q19_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p20_l_x), int(p20_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q20_l_x), int(q20_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p20_r_x), int(p20_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q20_r_x), int(q20_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p21_l_x), int(p21_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q21_l_x), int(q21_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p21_r_x), int(p21_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q21_r_x), int(q21_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p22_l_x), int(p22_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q22_l_x), int(q22_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p22_r_x), int(p22_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q22_r_x), int(q22_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p23_l_x), int(p23_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q23_l_x), int(q23_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p23_r_x), int(p23_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q23_r_x), int(q23_r_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p24_l_x), int(p24_l_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q24_l_x), int(q24_l_y)), 3, (255, 0, 0), -1)
    # cv2.circle(frame, (int(p24_r_x), int(p24_r_y)), 3, (0, 255, 255), -1)
    # cv2.circle(frame, (int(q24_r_x), int(q24_r_y)), 3, (255, 0, 0), -1)
    cv2.imshow('parts', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
