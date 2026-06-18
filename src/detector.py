import numpy as np
from ultralytics import YOLO
from src.config import MODEL_PATH, CONF_THRESHOLD, NMS_IOU_THRESHOLD, DETECTION_IMGSZ

class CardDetector:
    def __init__(self):
        try:
            self.model = YOLO(MODEL_PATH)
            print("Model loaded successfully. Class mapping:", self.model.names)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def detect(self, frame: np.ndarray):
        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            imgsz=DETECTION_IMGSZ,
            verbose=False,
        )
        return results[0] if results else None

    @staticmethod
    def preprocess_label(label: str) -> str:
        lbl = label.lower().strip()

        mapping = {
            'ace': 'A',
            'jack': 'J',
            'queen': 'Q',
            'king': 'K',
            'ten': '10'
        }

        if lbl in mapping:
            return mapping[lbl]

        if len(lbl) >= 2 and lbl[-1] in ('h', 'd', 'c', 's'):
            base = lbl[:-1].upper()
            if base in ('JACK', 'J'): return 'J'
            if base in ('QUEEN', 'Q'): return 'Q'
            if base in ('KING', 'K'): return 'K'
            if base in ('ACE', 'A'): return 'A'
            return base

        return lbl.upper()