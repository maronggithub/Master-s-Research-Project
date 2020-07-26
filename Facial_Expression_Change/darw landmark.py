import cv2
import dlib



PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
# Set camera resolution. The max resolution is webcam dependent
# so change it to a resolution that is both supported by your camera
# and compatible with your monitor
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# fullscreenset
cv2.namedWindow('landmark', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('landmark', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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
     # draw landmarks
     for i in range(shape.num_parts):
        p = shape.part(i)
        cv2.circle(frame, (p.x, p.y), 3, (0, 255, 255), -1)
    cv2.imshow('landmark', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()
