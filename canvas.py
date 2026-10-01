import cv2
import numpy as np


class CanvasManager:

    def __init__(self, brush_thickness=15, eraser_thickness=50):
        self.brush_thickness = brush_thickness
        self.eraser_thickness = eraser_thickness
        self.colors = [
            (255, 0, 255),  # Magenta
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 255, 255),  # Yellow
            (0, 0, 0),  # Eraser
        ]
        self.draw_color = self.colors[0]
        self.canvas = None
        self.xp, self.yp = 0, 0

    def init_canvas(self, frame_shape):
        """Creates an empty black canvas matching webcam frame dimensions."""
        h, w, _ = frame_shape
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def draw_header(self, frame):
        """Draws top toolbar with color options and clear canvas button."""
        cv2.rectangle(frame, (0, 0), (1280, 80), (40, 40, 40), cv2.FILLED)

        # Color Selector Boxes
        cv2.rectangle(frame, (20, 10), (120, 70), (255, 0, 255), cv2.FILLED)
        cv2.rectangle(frame, (140, 10), (240, 70), (255, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (260, 10), (360, 70), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (380, 10), (480, 70), (0, 255, 255), cv2.FILLED)

        # Eraser & Clear Buttons
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
        """Handles selection mode (2 fingers up) and drawing mode (1 finger up)."""
        if len(lm_list) < 21:
            self.xp, self.yp = 0, 0
            return frame

        x1, y1 = lm_list[8][1], lm_list[8][2]  # Index finger tip
        x2, y2 = lm_list[12][1], lm_list[12][2]  # Middle finger tip

        # Selection Mode: Index and Middle fingers up
        if fingers[1] and fingers[2]:
            self.xp, self.yp = 0, 0
            cv2.rectangle(
                frame, (x1, y1 - 25), (x2, y2 + 25), self.draw_color, cv2.FILLED
            )

            # Check header selections
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
                    self.draw_color = self.colors[4]  # Eraser
                elif 1100 < x1 < 1260:
                    self.canvas[:] = 0  # Clear Canvas

        # Drawing Mode: Only Index finger up
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
        """Blends live canvas drawing onto live webcam feed seamlessly."""
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, self.canvas)
        return frame