import json
import cv2
import torch
import torch.backends.cudnn as cudnn
from imutils.video import FPS
from flask import Flask, render_template, Response
from waitress import serve

from SSD.detect import infere
from SSD.fileVideoStream import FileVideoStream
from SSD.utils import voc_labels

# Config
app = Flask(__name__)
fvs = FileVideoStream(0).start()
fps = FPS().start()

counter = {}
for label in voc_labels:
    counter[label] = 0

detected_objects = []

# Inference device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = '../SSD/checkpoint/BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)  # Use map_location=torch.device('cpu') as 2nd parameter on laptop
model = checkpoint['model']
model = model.to(device)
model.eval()
cudnn.benchmark = True
cudnn.enabled = True


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    global detected_objects, counter
    while True:
        fps = FPS().start()
        frame = fvs.read()
        image, new_detected_objects, counter = infere(frame, counter, detected_objects, model)
        if len(new_detected_objects) != 0:
            detected_objects = new_detected_objects
        _, encodedImage = cv2.imencode('.jpg', image)
        fps.update()
        fps.stop()
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/getObjects', methods=['GET'])
def getObjects():
    return Response(json.dumps(counter), mimetype='application/json')


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=80)
