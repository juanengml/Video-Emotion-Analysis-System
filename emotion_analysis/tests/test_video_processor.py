import unittest
from unittest.mock import MagicMock
from emotion_analysis.vision.capture import VideoCapture
from emotion_analysis.vision.detection import EmotionDetector
from emotion_analysis.io.logging import EmotionEventLogger
from emotion_analysis.vision.processing import VideoProcessor
from emotion_analysis.vision.video_write import VideoWriter
from flask_opencv_streamer.streamer import Streamer


class TestVideoProcessor(unittest.TestCase):

    def setUp(self):
        self.video_source = '/home/verde003/Downloads/Exercicio emoção pessoas.mov'
        self.output_file = '/home/verde003/Downloads/output_model.mp4'
        self.detector_model = '/home/verde003/Modelos/camera-to-http/yolov8n-face.pt'

        self.capture = VideoCapture(self.video_source)
        self.detector = EmotionDetector(self.detector_model)
        self.logger = EmotionEventLogger()
        self.writer = VideoWriter(self.output_file, 640, 480, fps=20)
        self.streamer = Streamer(3031, False)

    def test_video_processor_initialization(self):
        processor = VideoProcessor(self.capture, self.detector, self.logger, self.writer, self.streamer)
        self.assertIsInstance(processor, VideoProcessor)

    def test_process_video(self):
        # Mock objects
        mock_capture = MagicMock(spec=VideoCapture)
        mock_detector = MagicMock(spec=EmotionDetector)
        mock_logger = MagicMock(spec=EmotionEventLogger)
        mock_writer = MagicMock(spec=VideoWriter)
        mock_streamer = MagicMock(spec=Streamer)

        # Initialize processor with mock objects
        processor = VideoProcessor(mock_capture, mock_detector, mock_logger, mock_writer, mock_streamer)

        # Call the process_video method
        processor.process_video()

        # Assert that relevant methods were called on mock objects
        mock_capture.capture.isOpened.assert_called()
        mock_capture.capture.read.assert_called()
        mock_detector.detect_faces.assert_called()
        mock_logger.log_event.assert_called()
        mock_writer.write_frame.assert_called()
        mock_streamer.update_frame.assert_called()

    def tearDown(self):
        # Clean up resources if needed
        pass


if __name__ == '__main__':
    unittest.main()
