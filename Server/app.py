import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from imutils.video import FPS
from torchvision import transforms
from flask import Flask, render_template, Response
from SSD.detect import infere
from SSD.fileVideoStream import FileVideoStream
from SSD.utils import voc_labels

# Config
app = Flask(__name__)
fvs = FileVideoStream(0, 1000000).start()
time.sleep(1.0)
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

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    global detected_objects, counter
    while True:
        fps = FPS().start()
        frame = fvs.read()
        image, detected_objects, counter = infere(frame, counter, detected_objects, normalize, to_tensor, model)
        _, encodedImage = cv2.imencode('.jpg', image)
        fps.update()
        fps.stop()
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
