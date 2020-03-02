import socket
import threading
import time
from stats import Stats
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import ffmpeg
from io import BytesIO
import os
import argparse
import json
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
from keras.backend import clear_session
import time

class Tello:
    def __init__(self):
        self.local_ip = ''
        self.local_port = 9000
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket.bind((self.local_ip, self.local_port))
        self.saved_frames = []
        self.tello_ip = '192.168.10.1'
        self.tello_port = 8889
        self.tello_adderss = (self.tello_ip, self.tello_port)
        self.log = []
        config_path  = 'config.json'
        with open(config_path) as config_buffer:    
            self.config = json.load(config_buffer)
        clear_session()
        self.infer_model = load_model(self.config['train']['saved_weights_name'],compile=False)
        self.infer_model._make_predict_function()
        self.net_h, self.net_w = 416, 416 # a multiple of 32, the smaller the faster
        self.obj_thresh, self.nms_thresh = 0.5, 0.45
        self.MAX_TIME_OUT = 10.0
        """
        #video socket
        self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for receiving video stream
        self.local_video_port = 11111  # port for receiving video stream
        """
        self.socket.sendto(b'streamon', self.tello_adderss)
        print ('sent: streamon')
        #self.socket_video.bind((self.local_ip, self.local_video_port))

        
        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        #self.bytIO = BytesIO()
        self.cap = cv2.VideoCapture('udp://127.0.0.1:11111')
        
        # thread for receiving video
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True
        self.receive_video_thread.start()
        
        
    def get_boxes(self):
        batch_size  = 1
        images      = []
        images += [self.frame]
        if (len(images)==batch_size) or (ret_val==False and len(images)>0):
            batch_boxes = get_yolo_boxes(self.infer_model, images, self.net_h, self.net_w, self.config['model']['anchors'], self.obj_thresh, self.nms_thresh)
            print("reading images: "+str(len(images)))
            for i in range(len(images)):
                draw_boxes(images[i], batch_boxes[i], self.config['model']['labels'], self.obj_thresh)
                cv2.imshow('video with bboxes', images[i])
            images = []
            if cv2.waitKey(1) == 27:
                cv2.destroyWindow('video with bboxes')
        

    def get_saved_boxes(self):
        batch_size  = 1
        images      = []
        images  = self.saved_frames
        count = []
        if (len(images)==batch_size) or (ret_val==False and len(images)>0):
            batch_boxes = get_yolo_boxes(self.infer_model, images, self.net_h, self.net_w, self.config['model']['anchors'], self.obj_thresh, self.nms_thresh)
            print("reading images: "+str(len(images)))
            for i in range(len(images)):
                _,cnt = draw_boxes(images[i], batch_boxes[i], self.config['model']['labels'], self.obj_thresh)
                count.append(cnt)
                cv2.imshow('video with bboxes', images[i])
                cv2.imwrite("static/video-" + str(i) + ".jpg", images[i])
            images = []
            if cv2.waitKey(1) == 27:
                cv2.destroyWindow('video with bboxes')
        return count 
    
    def send_command(self, command):
        """
        Send a command to the ip address. Will be blocked until
        the last command receives an 'OK'.
        If the command fails (either b/c time out or error),
        will try to resend the command
        :param command: (str) the command to send
        :param ip: (str) the ip of Tello
        :return: The latest command response
        """
        self.log.append(Stats(command, len(self.log)))

        self.socket.sendto(command.encode('utf-8'), self.tello_adderss)
        print('sending command: %s to %s'%(command, self.tello_ip))

        start = time.time()
        while not self.log[-1].got_response():
            now = time.time()
            diff = now - start
            if diff > self.MAX_TIME_OUT:
                print('Max timeout exceeded... command %s'%(command))
                # TODO: is timeout considered failure or next command still get executed
                # now, next one got executed
                return False
        print('Done!!! sent command: %s to %s'%(command, self.tello_ip))
        return True
    
    def _receive_thread(self):
        """Listen to responses from the Tello.

        Runs as a thread, sets self.response to whatever the Tello last returned.

        """
        while True:
            try:
                #self.response, ip = self.socket.recvfrom(1024)
                self.response, ip = self.socket.recvfrom(128)
                print('from %s: %s'%(ip, self.response))

                self.log[-1].add_response(self.response)
            except socket.error as exc:
                print("Caught exception socket.error : %s"%(exc))

    def save_frame(self):
        self.saved_frames += [self.frame]
        
    def receive_video(self):
        """
        Listens for video streaming (raw h264) from the Tello.

        Runs as a thread, sets self.frame to the most recent frame Tello captured.

        """
        self.cap = cv2.VideoCapture('udp://127.0.0.1:11111')
        if(self.cap.isOpened()):
          # Capture frame-by-frame
          ret, frame = self.cap.read()
          self.frame = frame

          if ret == True:
              cv2.imshow('original', frame)
              time.sleep(2)
              if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                return
              
              batch_size  = 1
              images      = []
              images += [frame]
              if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                  batch_boxes = get_yolo_boxes(self.infer_model, images, self.net_h, self.net_w, self.config['model']['anchors'], self.obj_thresh, self.nms_thresh)
                  print("reading images: "+str(len(images)))
                  for i in range(len(images)):
                      draw_boxes(images[i], batch_boxes[i], self.config['model']['labels'], self.obj_thresh)
                      cv2.imshow('video with bboxes', images[i])
                  images = []
               
          else:
            return

    def start_video_thread(self):
        self.cap = cv2.VideoCapture('udp://127.0.0.1:11111')
        # thread for receiving video
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True
        self.receive_video_thread.start()


    def stop_video_thread(self):
        self.receive_video_thread.stop()
        
    def _receive_video_thread(self):
        """
        Listens for video streaming (raw h264) from the Tello.

        Runs as a thread, sets self.frame to the most recent frame Tello captured.

        """
        
        while(self.cap.isOpened()):
          # Capture frame-by-frame
          ret, frame = self.cap.read()
          self.frame = frame
          
          if ret == True:
              cv2.imshow('original', frame)
              if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break  # esc to quit
              
        
                
    def on_close(self):
        self.socket.close()
        #self.socket_video.close()
        #cv2.destroyAllWindows()
        pass
        # for ip in self.tello_ip_list:
        #     self.socket.sendto('land'.encode('utf-8'), (ip, 8889))
        # self.socket.close()

    def get_log(self):
        return self.log

