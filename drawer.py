import cv2

class FrameDrawer:
    def __init__(self, config: dict):
        self.config = config

    def draw_detections(self, frame, detections: list):
        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["coords"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self.config["box_color"],
                self.config["box_thickness"]
            )

            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config["text_scale"],
                self.config["text_color"],
                self.config["text_thickness"]
            )

        return frame