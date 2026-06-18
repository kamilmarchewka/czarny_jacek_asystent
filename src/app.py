from collections import deque, Counter
import cv2
from src.config import BUFFER_SIZE, WEB_CAM_INDEX, MID_LINE_RATIO
from src.detector import CardDetector
from src.game_logic import BlackjackLogic
from src.ui_renderer import UIRenderer
from src.video_stream import VideoStream

DEDUP_DISTANCE = 80

def deduplicate_cards(card_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    unique: list[tuple[str, int]] = []
    for rank, cx in card_list:
        too_close = any(
            rank == ur and abs(cx - ux) < DEDUP_DISTANCE
            for ur, ux in unique
        )
        if not too_close:
            unique.append((rank, cx))
    return unique


class DetectionStabilizer:
    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self.player_buffer: deque = deque(maxlen=buffer_size)
        self.dealer_buffer: deque = deque(maxlen=buffer_size)

    def update_and_get_stable(
        self,
        current_player: list[tuple[str, int]],
        current_dealer: list[tuple[str, int]],
    ) -> tuple[list[str], list[str]]:
        player_sorted = tuple(r for r, _ in sorted(current_player, key=lambda c: c[1]))
        dealer_sorted = tuple(r for r, _ in sorted(current_dealer, key=lambda c: c[1]))

        self.player_buffer.append(player_sorted)
        self.dealer_buffer.append(dealer_sorted)

        if not self.player_buffer:
            return [], []

        stable_player = list(Counter(self.player_buffer).most_common(1)[0][0])
        stable_dealer = list(Counter(self.dealer_buffer).most_common(1)[0][0])

        return stable_player, stable_dealer


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
            mid_line = int(height * MID_LINE_RATIO)

            results = self.detector.detect(frame)

            raw_player_cards: list[tuple[str, int]] = []
            raw_dealer_cards: list[tuple[str, int]] = []

            if results and results.boxes:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    raw_label = self.detector.model.names[cls_id]
                    rank = self.detector.preprocess_label(raw_label)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    if center_y > mid_line:
                        raw_player_cards.append((rank, center_x))
                    else:
                        raw_dealer_cards.append((rank, center_x))

                    self.ui.draw_card_box(frame, (x1, y1, x2, y2), raw_label, conf)

            raw_player_cards = deduplicate_cards(raw_player_cards)
            raw_dealer_cards = deduplicate_cards(raw_dealer_cards)

            stable_player, stable_dealer = self.stabilizer.update_and_get_stable(
                raw_player_cards, raw_dealer_cards
            )

            p_score, _ = BlackjackLogic.calculate_score_and_is_soft(stable_player)
            d_score, _ = BlackjackLogic.calculate_score_and_is_soft(stable_dealer)
            hint = BlackjackLogic.get_hint(stable_player, stable_dealer)

            self.ui.draw_interface(
                frame, p_score, d_score, hint, stable_player, stable_dealer, mid_line
            )

            cv2.imshow("Blackjack AI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.vs.release()
        cv2.destroyAllWindows()