import bbox_visualizer as bbv
from datetime import datetime as dt
import cv2

class VideoProcessor:
    def __init__(self, capture, detector, logger, writer, streamer):
        self.capture = capture
        self.detector = detector
        self.logger = logger
        self.writer = writer
        self.streamer = streamer

    def process_frame(self, frame):
        bboxes = self.detector.detect_faces(frame)
        face_emotions = []

        for bbox in bboxes:
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
            emotion = self.detector.analyze_emotion(face_roi)
            face_emotions.append(emotion)
            timestamp = dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            self.logger.log_event(timestamp, emotion, bbox)

            frame = bbv.draw_rectangle(frame, (x, y, x + w, y + h), is_opaque=True)
            frame = bbv.add_label(frame, emotion, (x, y, x + w, y + h), top=False)

        predominant_emotion = max(set(face_emotions), key=face_emotions.count) if face_emotions else "None"
        frame = self.draw_table_on_image(frame, len(face_emotions), predominant_emotion)
        return frame

    def draw_table_on_image(self, image, total_faces, predominant_emotion):
        table_info = f"   [ Total Rostos: {total_faces} |  Sentimento Predominante: {predominant_emotion}  | time: {dt.now()} ]"
        x, y = 20, 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        color = (0, 255, 127)
        line_type = cv2.LINE_AA
        cv2.putText(image, table_info, (x, y), font, font_scale, color, font_thickness, line_type)
        return image

    def process_video(self):
        while True:
            ret, frame = self.capture.read_frame()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            self.writer.write_frame(processed_frame)
            self.streamer.update_frame(processed_frame)

            if not self.streamer.is_streaming:
                self.streamer.start_streaming()

            cv2.waitKey(30)

        self.capture.release()
        self.writer.release()
