from flask import Flask, render_template, send_from_directory, Response
from flask_socketio import SocketIO
import cv2
from cv_controller import get_frame, get_frame_with_hands_detected

app = Flask(__name__, static_folder='..', static_url_path='')
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return send_from_directory('..', 'index.html')

@app.route('/mjpeg')
def stream():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame') # Response to stream output of generator function

@app.route('/annotated')
def stream_with_annotations():
    return Response(get_frame_with_hands_detected(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('startWebcam')
def start(data):
    print('received message: ' + data)

if __name__ == '__main__':
    socketio.run(app, port=8000)

# have to see what stuff to change when running with gunicorn like port = 8000


# TODO
# make new div and styling for webcam
# remove 2 player functionality
# start/stop camera button, off by default
# updated instructions and visible by default with local storage to toggle default on
# then detect two pens when camera is on
# overlay a circle between them at the correct depth
# then have a start race button and countdown to begin it
# then implement the processing loop for distance between them (I think) and turn angle
# send these to game and have these control the car