import cv2
import mediapipe as mp

# fmt: off
# type: ignore
from mediapipe.python.solutions import drawing_utils as mp_draw  # type: ignore
from mediapipe.python.solutions import hands as mp_hands          # type: ignore
# fmt: on


class HandTracker:

    def __init__(
        self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # MediaPipe solution references
        self.mp_hands = mp_hands
        self.mp_draw = mp_draw

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    def find_hands(self, frame, draw=True):
        """Processes RGB frame and draws hand skeleton landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )
        return frame

    def get_landmark_positions(self, frame):
        """Extracts (x, y) coordinates for all 21 hand landmarks in pixel space."""
        lm_list = []
        if (
            hasattr(self, "results")
            and self.results
            and self.results.multi_hand_landmarks
        ):
            my_hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            for id, lm in enumerate(my_hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list

    def fingers_up(self, lm_list):
        """Returns list of 5 booleans [Thumb, Index, Middle, Ring, Pinky] indicating extension."""
        fingers = []
        if len(lm_list) < 21:
            return fingers

        # Thumb (horizontal check)
        if lm_list[self.tip_ids[0]][1] < lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers (vertical check)
        for id in range(1, 5):
            if (
                lm_list[self.tip_ids[id]][2]
                < lm_list[self.tip_ids[id] - 2][2]
            ):
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers