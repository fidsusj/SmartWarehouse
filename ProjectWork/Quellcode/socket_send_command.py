self.tello_address = ('192.168.10.1', 8889)
self.log = []

def send_command(self, command):
	self.log.append(Stats(command, len(self.log)))
    self.socket.sendto(command.encode('utf-8'), self.tello_address)

    start = time.time()
    while not self.log[-1].got_response():
        now = time.time()
        diff = now - start
        if diff > self.MAX_TIME_OUT:
            return False
    return True
		
def receive_thread(self):
    while True:
        try:
            self.response, ip = self.socket.recvfrom(128)
            self.log[-1].set_response(self.response)
        except socket.error as exc:
            print("Caught exception socket.error : %s"%(exc))