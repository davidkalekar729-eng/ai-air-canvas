import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


# ==========================================
# 1. PROCEDURAL SPIDER-MAN MASK MODULE
# ==========================================
class SpiderManMask:

    def __init__(self):
        # Local XML fallback file path
        self.cascade_filename = "haarcascade_frontalface_default.xml"
        self._ensure_cascade_exists()

        self.face_cascade = cv2.CascadeClassifier(self.cascade_filename)
        if self.face_cascade.empty():
            print("Warning: Face cascade failed to load.")

    def _ensure_cascade_exists(self):
        """Downloads the official Haar cascade XML if missing locally."""
        if not os.path.exists(self.cascade_filename):
            print("Downloading face detection model file...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                urllib.request.urlretrieve(url, self.cascade_filename)
                print("Model file downloaded successfully.")
            except Exception as e:
                print(f"Error downloading cascade file: {e}")

    def overlay_mask(self, frame):
        if self.face_cascade.empty():
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
        )

        for x, y, w, h in faces:
            center_x, center_y = x + w // 2, y + h // 2
            axes = (int(w * 0.55), int(h * 0.65))

            # Create overlay mask layer
            overlay = frame.copy()

            # 1. Red Base Mask
            cv2.ellipse(
                overlay,
                (center_x, center_y),
                axes,
                0,
                0,
                360,
                (0, 0, 200),
                -1,
            )

            # 2. Black Web Lines (Radial & Concentric)
            for angle in range(0, 360, 30):
                rad = np.radians(angle)
                end_x = int(center_x + axes[0] * np.cos(rad))
                end_y = int(center_y + axes[1] * np.sin(rad))
                cv2.line(
                    overlay,
                    (center_x, center_y),
                    (end_x, end_y),
                    (20, 20, 20),
                    2,
                )

            for scale in [0.3, 0.6, 0.85]:
                cv2.ellipse(
                    overlay,
                    (center_x, center_y),
                    (int(axes[0] * scale), int(axes[1] * scale)),
                    0,
                    0,
                    360,
                    (20, 20, 20),
                    2,
                )

            # 3. Spider-Man Eyes
            eye_w, eye_h = int(w * 0.18), int(h * 0.12)
            left_eye_center = (
                center_x - int(w * 0.22),
                center_y - int(h * 0.08),
            )
            right_eye_center = (
                center_x + int(w * 0.22),
                center_y - int(h * 0.08),
            )

            # Outer Black Borders
            cv2.ellipse(
                overlay,
                left_eye_center,
                (eye_w + 6, eye_h + 6),
                -15,
                0,
                360,
                (0, 0, 0),
                -1,
            )
            cv2.ellipse(
                overlay,
                right_eye_center,
                (eye_w + 6, eye_h + 6),
                15,
                0,
                360,
                (0, 0, 0),
                -1,
            )

            # Inner White Lenses
            cv2.ellipse(
                overlay,
                left_eye_center,
                (eye_w, eye_h),
                -15,
                0,
                360,
                (255, 255, 255),
                -1,
            )
            cv2.ellipse(
                overlay,
                right_eye_center,
                (eye_w, eye_h),
                15,
                0,
                360,
                (255, 255, 255),
                -1,
            )

            # Blend onto frame
            alpha = 0.75
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame


# ==========================================
# 2. CANVAS MANAGER MODULE
# ==========================================
class CanvasManager:

    def __init__(self, brush_thickness=15, eraser_thickness=50):
        self.brush_thickness = brush_thickness
        self.eraser_thickness = eraser_thickness
        self.colors = [
            (255, 0, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 255, 255),
            (0, 0, 0),
        ]
        self.draw_color = self.colors[0]
        self.canvas = None
        self.xp, self.yp = 0, 0

    def init_canvas(self, frame_shape):
        h, w, _ = frame_shape
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def draw_header(self, frame):
        cv2.rectangle(frame, (0, 0), (1280, 80), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(frame, (20, 10), (120, 70), (255, 0, 255), cv2.FILLED)
        cv2.rectangle(frame, (140, 10), (240, 70), (255, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (260, 10), (360, 70), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (380, 10), (480, 70), (0, 255, 255), cv2.FILLED)
        cv2.rectangle(frame, (500, 10), (620, 70), (200, 200, 200), cv2.FILLED)
        cv2.putText(
            frame,
            "ERASER",
            (510, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.rectangle(frame, (1100, 10), (1260, 70), (0, 0, 255), cv2.FILLED)
        cv2.putText(
            frame,
            "CLEAR",
            (1130, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return frame

    def process_drawing(self, frame, fingers, lm_list):
        if not lm_list or len(lm_list) < 21:
            self.xp, self.yp = 0, 0
            return frame

        x1, y1 = lm_list[8][0], lm_list[8][1]
        x2, y2 = lm_list[12][0], lm_list[12][1]

        if fingers[1] and fingers[2]:
            self.xp, self.yp = 0, 0
            cv2.rectangle(
                frame, (x1, y1 - 25), (x2, y2 + 25), self.draw_color, cv2.FILLED
            )

            if y1 < 80:
                if 20 < x1 < 120:
                    self.draw_color = self.colors[0]
                elif 140 < x1 < 240:
                    self.draw_color = self.colors[1]
                elif 260 < x1 < 360:
                    self.draw_color = self.colors[2]
                elif 380 < x1 < 480:
                    self.draw_color = self.colors[3]
                elif 500 < x1 < 620:
                    self.draw_color = self.colors[4]
                elif 1100 < x1 < 1260:
                    self.canvas[:] = 0

        elif fingers[1] and not fingers[2]:
            cv2.circle(frame, (x1, y1), 15, self.draw_color, cv2.FILLED)

            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x1, y1

            thickness = (
                self.eraser_thickness
                if self.draw_color == (0, 0, 0)
                else self.brush_thickness
            )

            cv2.line(
                self.canvas,
                (self.xp, self.yp),
                (x1, y1),
                self.draw_color,
                thickness,
            )
            self.xp, self.yp = x1, y1

        else:
            self.xp, self.yp = 0, 0

        return frame

    def merge_canvas(self, frame):
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, self.canvas)
        return frame


# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8, maxHands=1)
    canvas = CanvasManager()
    spidey_mask = SpiderManMask()

    print("AI Air Canvas Starting...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        canvas.init_canvas(frame.shape)

        # 1. Overlay Mask on Face
        frame = spidey_mask.overlay_mask(frame)

        # 2. Track Hand & Draw
        hands, frame = detector.findHands(frame, draw=True)

        if hands:
            hand = hands[0]
            lm_list = hand["lmList"]
            fingers = detector.fingersUp(hand)
            frame = canvas.process_drawing(frame, fingers, lm_list)

        frame = canvas.draw_header(frame)
        frame = canvas.merge_canvas(frame)

        cv2.imshow("AI Air Canvas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 