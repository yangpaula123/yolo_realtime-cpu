import cv2
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class CameraCapture:
    def __init__(self, camera_index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (索引: {camera_index})")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()