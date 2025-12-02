import cv2
import time

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