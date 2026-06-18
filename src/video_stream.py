import threading
import cv2

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.ret, self.frame = self.stream.read()

    def start(self):
        thread = threading.Thread(target=self.update, args=(), daemon=True)
        thread.start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream.isOpened():
                self.ret, self.frame = self.stream.read()
            else:
                self.stopped = True

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.stream.release()