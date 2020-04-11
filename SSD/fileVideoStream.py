from threading import Thread
import cv2


class FileVideoStream:
    def __init__(self, path):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.buffer = None

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def stop(self):
        self.stopped = True

    def update(self):
        while True:
            if self.stopped:
                return
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                self.stop()
                return
            self.buffer = frame

    def read(self):
        return self.buffer
