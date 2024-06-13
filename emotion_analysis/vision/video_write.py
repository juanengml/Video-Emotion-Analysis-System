import cv2

class VideoWriter:
    def __init__(self, output_file, frame_width, frame_height, fps=20):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    def write_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
