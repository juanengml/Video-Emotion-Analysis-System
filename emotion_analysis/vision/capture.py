import cv2

class VideoCapture:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise IOError("Cannot open video stream")

    def read_frame(self):
        ret, frame = self.capture.read()
        return ret, frame

    def release(self):
        self.capture.release()
