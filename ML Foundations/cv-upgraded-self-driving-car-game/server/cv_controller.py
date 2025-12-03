import cv2
import time
import mediapipe as mp
import numpy as np

# mediapipe setup from docs
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# setup camera and resolution
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def get_frame():
    while 1 == 1: # my brother likes this instead of True
        success, image = camera.read()
        if not success:
            continue
        unmirrored_image = cv2.flip(image, 1) # 1 means horizontal axis
        _, frame = cv2.imencode('.jpg', unmirrored_image)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n') # generator function
        # ^^ frame = boundary marker, then indicate this is a jpeg image, bytestring, and end marker

def get_frame_with_hands_detected():
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while camera.isOpened():
            success, image = camera.read()
            if not success:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pts = []
            h, w, _ = image.shape
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark = hand_landmarks.landmark[5]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    tupl = (x, y)
                    pts.append(tupl)
            for pt in pts:
                cv2.circle(image, pt, 5, (0, 255, 0), -1)

            pts = sorted(pts, key = lambda x: x[0]) # ensure the user's left hand is the second keypoint to keep angle calculation consistent
            angle_deg = 0

            if len(pts) == 2:
                cv2.line(image, (pts[1]), (0, pts[1][1]), (0, 255, 0), thickness=3, lineType=8)
                cv2.line(image, (pts[0]), (pts[1]), (255, 0, 0), thickness=3, lineType=8)
            
                right_angle_pt = (pts[0][0], pts[1][1])

                opposite = pts[1][1] - pts[0][1]
                adjacent = pts[1][0] - pts[0][0]
                try:
                    angle = np.arctan(opposite / adjacent)
                except Exception as e:
                    pass # this means user's hands are stacked vertically to make adjacent = 0, we don't have to do anything besides wait for them to move back
                else:
                    angle_deg = angle * 180 / np.pi


            unmirrored_image = cv2.flip(image, 1) 
            unmirrored_image = cv2.putText(unmirrored_image, str(int(angle_deg)) + ' deg', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 0, 0), 2, cv2.LINE_AA)
            
            _, frame = cv2.imencode('.jpg', unmirrored_image)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
