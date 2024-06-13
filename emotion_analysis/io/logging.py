class EmotionEventLogger:
    def __init__(self):
        self.emotion_events = []

    def log_event(self, timestamp, emotion, bbox):
        self.emotion_events.append({
            'timestamp': timestamp,
            'emotion': emotion,
            'bbox': bbox
        })

    def get_events(self):
        return self.emotion_events
