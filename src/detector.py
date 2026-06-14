from ultralytics import YOLO
from src.config import MODEL_PATH, CONF_THRESHOLD, NMS_IOU_THRESHOLD

class CardDetector:
    def __init__(self):
        try:
            self.model = YOLO(MODEL_PATH)
            print("Model loaded successfully. Class mapping:", self.model.names)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def detect(self, frame):
        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            verbose=False
        )
        return results[0] if results else None

    def preprocess_label(self, label):
        if len(label) > 1 and label[-1].lower() in ['h', 'd', 'c', 's']:
            return label[:-1].upper()
        return label.upper()