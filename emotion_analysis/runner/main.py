from emotion_analysis.vision.capture import VideoCapture
from emotion_analysis.vision.detection import EmotionDetector
from emotion_analysis.io.logging import EmotionEventLogger
from emotion_analysis.vision.processing import VideoProcessor
from emotion_analysis.vision.video_write import VideoWriter
from flask_opencv_streamer.streamer import Streamer

import cv2


def main():
    video_source = '/home/verde003/Downloads/Exercicio emoção pessoas.mov'
    output_file = "/home/verde003/Downloads/output_model.mp4"

    capture = VideoCapture(video_source)
    frame_width = int(capture.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = EmotionDetector('/home/verde003/Modelos/camera-to-http/yolov8n-face.pt')
    logger = EmotionEventLogger()
    writer = VideoWriter(output_file, frame_width, frame_height, fps=20)
    streamer = Streamer(3031, False)

    processor = VideoProcessor(capture, detector, logger, writer, streamer)
    processor.process_video()

    for event in logger.get_events():
        print(f"Timestamp: {event['timestamp']} - Emotion: {event['emotion']}")

if __name__ == "__main__":
    main()
