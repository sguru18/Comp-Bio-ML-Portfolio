import cv2
import time
import mediapipe as mp

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

        while 1 == 1:
            success, image = camera.read()
            if not success:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            unmirrored_image = cv2.flip(image, 1) 
            _, frame = cv2.imencode('.jpg', unmirrored_image)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
