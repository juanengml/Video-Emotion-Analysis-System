from ultralytics import YOLO
from deepface import DeepFace

class EmotionDetector:
    def __init__(self, model_path, conf_threshold=0.55, img_size=1280, max_det=1000):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.max_det = max_det

    def detect_faces(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, imgsz=self.img_size, max_det=self.max_det)
        bboxes = [map(int, bbox[:4]) for result in results for bbox in result.boxes.xyxy]
        return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in bboxes]

    def analyze_emotion(self, face_roi):
        try:
            analyze = DeepFace.analyze(face_roi, actions=['emotion'])
            return analyze[0]['dominant_emotion']
        except:
            return "Face"
