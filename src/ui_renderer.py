import cv2

class UIRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_card_box(self, frame, box, label, conf):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        display_msg = f"{label} {conf:.2f}"
        cv2.putText(frame, display_msg, (x1, y1 - 10), self.font, 0.5, (255, 255, 0), 2)

    def draw_interface(self, frame, p_score, d_score, hint, stable_player, stable_dealer, mid_line):
        height, width, _ = frame.shape

        cv2.line(frame, (0, mid_line), (width, mid_line), (255, 255, 255), 1)

        cv2.text_size = cv2.putText(frame, f"Dealer: {d_score} ({', '.join(stable_dealer)})", (10, 30), self.font, 0.8,
                                    (0, 255, 255), 2)
        cv2.putText(frame, f"Player: {p_score} ({', '.join(stable_player)})", (10, mid_line + 30), self.font, 0.8,
                    (0, 255, 255), 2)

        color_hint = (0, 255, 0)
        if hint == "HIT":
            color_hint = (0, 0, 255)
        elif hint in ["BUST", "WAIT"]:
            color_hint = (0, 0, 139)

        cv2.rectangle(frame, (width - 240, height - 100), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 240, height - 100), (width - 10, height - 10), color_hint, 3)
        cv2.putText(frame, hint, (width - 210, height - 40), self.font, 1.5, color_hint, 4)