import json
import os
import threading
import cv2
from imutils.video import FPS
from flask import Flask, render_template, Response
from waitress import serve
from Drone.tello import Tello
import keyboard as keyboard
import YOLO.build.darknet.x64.darknet_video as YOLO

# Config
app = Flask(__name__)
tello = Tello('', 8889)


@app.route('/')
def index():
    return render_template('index.html')


def flight_sequence():
    tello.send_command('takeoff')
    tello.send_command('up 150')
    tello.send_command('right 100')
    tello.send_command('down 100')
    tello.send_command('left 100')
    tello.send_command('up 100')
    tello.send_command('right 50')
    tello.send_command('back 150')
    tello.send_command('streamoff')
    tello.send_command('land')


def gen():
    global detected_objects, counter
    while True:
        frame = tello.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps = FPS().start()
            frame = YOLO.infere(frame)
            fps.update()
            fps.stop()
            if keyboard.is_pressed('s'):
                os.chdir("C:\\Users\\Felix\\OneDrive\\Desktop\\Saved")
                cv2.imwrite("picture.jpeg", frame)
            _, encodedImage = cv2.imencode('.jpg', frame)
            # print("[INFO] approx. FPS: {:.2f}".format(fps.fps())) # Comment in when inference integrated
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fly')
def fly():
    flight_thread = threading.Thread(target=flight_sequence)
    flight_thread.start()


@app.route('/getObjects', methods=['GET'])
def getObjects():
    return Response(json.dumps(counter), mimetype='application/json')


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=80)
