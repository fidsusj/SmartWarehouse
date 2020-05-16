import socket
import threading
from Drone.stats import Stats
import time


class Tello:
    def __init__(self):
        # Local Address
        self.local_ip = ''
        self.local_port = 9000
        self.local_video_port = 11111

        # socket for sending cmd
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.local_ip, self.local_port))

        # socket for receiving video stream
        self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_video.bind((self.local_ip, self.local_video_port))

        # Tello Address
        self.tello_ip = '192.168.10.1'
        self.tello_port = 8889
        self.tello_address = (self.tello_ip, self.tello_port)
        self.socket.sendto(b'streamon', self.tello_address)

        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()

        # thread for receiving video ack
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True
        self.receive_video_thread.start()

        self.log = []
        self.MAX_TIME_OUT = 10.0

    def send_command(self, command):
        """
        Send a command to the ip address. Will be blocked until the last command receives an 'OK'.
        If the command fails (either b/c time out or error), will try to resend the command
        :param command: (str) the command to send
        :param ip: (str) the ip of Tello
        :return: The latest command response
        """
        self.log.append(Stats(command, len(self.log)))

        self.socket.sendto(command.encode('utf-8'), self.tello_address)
        print('sending command: %s to %s' % (command, self.tello_ip))

        start = time.time()
        while not self.log[-1].got_response():
            now = time.time()
            diff = now - start
            if diff > self.MAX_TIME_OUT:
                print('Max timeout exceeded... command %s' % (command))
                # TODO: is timeout considered failure or next command still get executed
                # now, next one got executed
                return False
        print('Done!!! sent command: %s to %s' % (command, self.tello_ip))
        return True

    def _receive_thread(self):
        while True:
            try:
                # self.response, ip = self.socket.recvfrom(1024)
                self.response, ip = self.socket.recvfrom(128)
                print('from %s: %s' % (ip, self.response))

                self.log[-1].add_response(self.response)
            except socket.error as exc:
                print("Caught exception socket.error : %s" % (exc))

    def _receive_video_thread(self):
        while True:
            try:
                # self.response, ip = self.socket_video.recvfrom(1024)
                self.response, ip = self.socket_video.recvfrom(128)
                print('from %s: %s' % (ip, self.response))

                self.log[-1].add_response(self.response)
            except socket.error as exc:
                print("Caught exception socket.error : %s" % (exc))

    def on_close(self):
        self.socket.close()
        self.socket_video.close()