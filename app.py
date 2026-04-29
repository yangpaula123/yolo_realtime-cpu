import cv2
import logging
from ultralytics import YOLO
from config import CONFIG
from detector import YOLODetector
from camera import CameraCapture
from drawer import FrameDrawer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    detector = YOLODetector(CONFIG)
    camera = CameraCapture(
        CONFIG["camera_index"],
        CONFIG["frame_width"],
        CONFIG["frame_height"]
    )
    drawer = FrameDrawer(CONFIG)

    frame_count = 0
    last_detections = []

    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break

        frame_count += 1

        if frame_count % CONFIG["detection_interval"] == 0:
            results = detector.detect(frame)
            last_detections = detector.get_detections(results)

        display_frame = drawer.draw_detections(frame.copy(), last_detections)

        cv2.imshow("YOLOv8 Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

