import cv2
from ultralytics import YOLO
from collections import Counter, deque

MODEL_PATH = 'data/best.pt'
BUFFER_SIZE = 8
CONF_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.3
WEB_CAM_INDEX = 0

class Card:
    def __init__(self, rank):
        self.rank = rank

    def get_value(self):
        if self.rank in ['J', 'Q', 'K', '10']: return 10
        if self.rank == 'A': return 11
        return int(self.rank)


class BlackjackLogic:
    @staticmethod
    def calculate_score(ranks):
        if not ranks: return 0
        score = sum(Card(r).get_value() for r in ranks)
        aces = ranks.count('A')
        while score > 21 and aces > 0:
            score -= 10
            aces -= 1
        return score

    @staticmethod
    def get_hint(p_score, d_up_card_val):
        if p_score == 0: return "WAIT"
        if p_score > 21: return "BUST"
        if p_score >= 17: return "STAND"
        if p_score <= 11: return "HIT"
        if d_up_card_val >= 2 and d_up_card_val <= 6:
            return "STAND"
        return "HIT"


class DetectionStabilizer:
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.player_buffer = deque(maxlen=buffer_size)
        self.dealer_buffer = deque(maxlen=buffer_size)

    def update_and_get_stable(self, current_player_cards, current_dealer_cards):
        self.player_buffer.append(tuple(sorted(current_player_cards)))
        self.dealer_buffer.append(tuple(sorted(current_dealer_cards)))

        if not self.player_buffer: return [], []

        stable_player = Counter(self.player_buffer).most_common(1)[0][0]
        stable_dealer = Counter(self.dealer_buffer).most_common(1)[0][0]

        return list(stable_player), list(stable_dealer)


class BlackjackAIApp:
    def __init__(self):
        print(f"Ładowanie modelu YOLO z {MODEL_PATH}...")
        try:
            self.model = YOLO(MODEL_PATH)
            print("Model załadowany. Mapowanie klas:", self.model.names)
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}")
            exit()

        self.stabilizer = DetectionStabilizer()
        self.cap = cv2.VideoCapture(WEB_CAM_INDEX)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def preprocess_label(self, label):
        if len(label) > 1 and label[-1].lower() in ['h', 'd', 'c', 's']:
            return label[:-1].upper()
        return label.upper()

    def run(self):
        print("Uruchamianie pętli wideo. Naciśnij 'q' aby wyjść.")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            height, width, _ = frame.shape
            mid_line = height // 2

            results = self.model(frame,
                                 conf=CONF_THRESHOLD,
                                 iou=NMS_IOU_THRESHOLD,
                                 verbose=False)[0]

            raw_player_cards = []
            raw_dealer_cards = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                raw_label = self.model.names[cls_id]
                rank = self.preprocess_label(raw_label)

                center_y = (y1 + y2) // 2
                if center_y > mid_line:
                    raw_player_cards.append(rank)
                else:
                    raw_dealer_cards.append(rank)

                color_box = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)

                display_msg = f"{raw_label} {conf:.2f}"
                cv2.putText(frame, display_msg, (x1, y1 - 10), self.font, 0.5, (255, 255, 0), 2)

            stable_player, stable_dealer = self.stabilizer.update_and_get_stable(raw_player_cards, raw_dealer_cards)

            p_score = BlackjackLogic.calculate_score(stable_player)
            d_score = BlackjackLogic.calculate_score(stable_dealer)

            d_up_val = 0
            if stable_dealer:
                d_up_val = Card(stable_dealer[0]).get_value()

            hint = BlackjackLogic.get_hint(p_score, d_up_val)

            self.draw_interface(frame, p_score, d_score, hint, stable_player, stable_dealer, mid_line, width, height)

            cv2.imshow('Blackjack AI Overlap Fix', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

    def draw_interface(self, frame, p_score, d_score, hint, stable_player, stable_dealer, mid_line, width, height):
        cv2.line(frame, (0, mid_line), (width, mid_line), (255, 255, 255), 1)

        cv2.putText(frame, f"Dealer: {d_score} ({', '.join(stable_dealer)})", (10, 30), self.font, 0.8, (0, 255, 255),
                    2)
        cv2.putText(frame, f"Player: {p_score} ({', '.join(stable_player)})", (10, mid_line + 30), self.font, 0.8,
                    (0, 255, 255), 2)

        color_hint = (0, 255, 0)
        if hint == "HIT": color_hint = (0, 0, 255)
        if hint == "BUST": color_hint = (0, 0, 139)

        cv2.rectangle(frame, (width - 240, height - 100), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 240, height - 100), (width - 10, height - 10), color_hint, 3)
        cv2.putText(frame, hint, (width - 210, height - 40), self.font, 1.5, color_hint, 4)


if __name__ == "__main__":
    app = BlackjackAIApp()
    app.run()