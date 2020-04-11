import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)
fvs = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        _, frame = fvs.read()
        _, encodedImage = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
