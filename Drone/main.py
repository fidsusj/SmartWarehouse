from flask import Flask,render_template
from Drone.basicTello import basicTello
from datetime import datetime
import time
from Drone.tello import Tello
import sys

app = Flask(__name__)


def connect():
    tello = basicTello()
    res = tello.send_command('command')
    print(res)

def fly():
    tello = Tello()
    res = tello.send_command('takeoff')
    # tello.send_command('up 20')
    # tello.send_command('forward 20')
    # res = tello.send_command('right 40')
    tello.save_frame()
    # tello.send_command('left 150')
    tello.send_command('land')
    ans = tello.get_saved_boxes()
    return ans

@app.route("/", methods=['GET', 'POST'])
def home():
    connect()
    #boxes = fly()
    time.sleep(5)
    src = 'C:\\Users\\I518134\\work\\drone\\Tello-Python3\\Simple-tello-control-GUI\\static\\video-0.jpg'
    return render_template('res_html.html', count=boxes, src=src)

if __name__ == "__main__":
    app.run(debug=True)
