from collections import deque, Counter
import cv2
from src.config import BUFFER_SIZE, WEB_CAM_INDEX
from src.detector import CardDetector
from src.game_logic import BlackjackLogic
from src.ui_renderer import UIRenderer
from src.video_stream import VideoStream

class DetectionStabilizer:
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.player_buffer = deque(maxlen=buffer_size)
        self.dealer_buffer = deque(maxlen=buffer_size)

    def update_and_get_stable(self, current_player_cards, current_dealer_cards):
        self.player_buffer.append(tuple(sorted(current_player_cards)))
        self.dealer_buffer.append(tuple(sorted(current_dealer_cards)))

        if not self.player_buffer:
            return [], []

        stable_player = Counter(self.player_buffer).most_common(1)[0][0]
        stable_dealer = Counter(self.dealer_buffer).most_common(1)[0][0]

        return list(stable_player), list(stable_dealer)


class BlackjackAIApp:
    def __init__(self):
        self.detector = CardDetector()
        self.ui = UIRenderer()
        self.stabilizer = DetectionStabilizer()
        self.vs = VideoStream(src=WEB_CAM_INDEX).start()

    def run(self):
        print("Starting video loop. Press 'q' to exit.")

        while True:
            ret, frame = self.vs.read()
            if not ret or frame is None:
                continue

            height, width, _ = frame.shape
            mid_line = height // 2

            results = self.detector.detect(frame)

            raw_player_cards = []
            raw_dealer_cards = []

            if results and results.boxes:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    raw_label = self.detector.model.names[cls_id]
                    rank = self.detector.preprocess_label(raw_label)

                    center_y = (y1 + y2) // 2
                    if center_y > mid_line:
                        raw_player_cards.append(rank)
                    else:
                        raw_dealer_cards.append(rank)

                    self.ui.draw_card_box(frame, (x1, y1, x2, y2), raw_label, conf)

            stable_player, stable_dealer = self.stabilizer.update_and_get_stable(raw_player_cards, raw_dealer_cards)

            p_score, _ = BlackjackLogic.calculate_score_and_is_soft(stable_player)
            d_score, _ = BlackjackLogic.calculate_score_and_is_soft(stable_dealer)
            hint = BlackjackLogic.get_hint(stable_player, stable_dealer)

            self.ui.draw_interface(frame, p_score, d_score, hint, stable_player, stable_dealer, mid_line)

            cv2.imshow('Blackjack AI Overlap Fix', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.vs.release()
        cv2.destroyAllWindows()