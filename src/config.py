from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = str(BASE_DIR / "data" / "best.pt")
BUFFER_SIZE = 8
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.3
WEB_CAM_INDEX = 0
MID_LINE_RATIO = 0.5
DETECTION_IMGSZ = 640