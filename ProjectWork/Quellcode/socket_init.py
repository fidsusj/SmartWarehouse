self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
self.socket.bind(('localhost', 9000))