import cv2
import logging
from ultralytics import YOLO
from typing import Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#  配置参数（便于修改）
CONFIG = {
    "model_name": "yolov8n.pt",
    "frame_width": 640,
    "frame_height": 480,
    "detection_interval": 2,  # 每N帧检测一次
    "target_classes": ["person", "cell phone", "book"],
    "box_color": (0, 255, 0),
    "box_thickness": 2,
    "text_color": (0, 255, 0),
    "text_scale": 0.6,
    "text_thickness": 2,
    "conf_threshold": 0.3,
    "camera_index": 0,
}


class YOLODetector:
    """YOLO实时检测器"""
    
    def __init__(self, config: dict):
        """初始化检测器
        
        Args:
            config: 配置字典
            
        Raises:
            RuntimeError: 模型加载失败
        """
        self.config = config
        self.model = None
        self.target_class_names = set(config["target_classes"])
        self._init_model()
    
    def _init_model(self):
        """加载并初始化YOLO模型"""
        try:
            logger.info(f"加载模型: {self.config['model_name']}")
            self.model = YOLO(self.config["model_name"])
            self.model.to("cpu")
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载模型: {e}")
    
    def detect(self, frame) -> Optional:
        """执行目标检测
        
        Args:
            frame: 输入图像
            
        Returns:
            YOLO检测结果对象
        """
        if self.model is None:
            return None
        
        results = self.model(frame, verbose=False)
        return results
    
    def get_detections(self, results) -> list:
        """提取目标检测结果
        
        Args:
            results: YOLO检测结果
            
        Returns:
            (label, confidence, box_coords) 的列表
        """
        detections = []
        
        if results is None:
            return detections
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                
                # 过滤低置信度
                if conf < self.config["conf_threshold"]:
                    continue
                
                # 只保留目标类别
                if label not in self.target_class_names:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "coords": (x1, y1, x2, y2)
                })
        
        return detections


class CameraCapture:
    """摄像头捕获器"""
    
    def __init__(self, camera_index: int, width: int, height: int):
        """初始化摄像头
        
        Args:
            camera_index: 摄像头索引
            width: 目标帧宽度
            height: 目标帧高度
            
        Raises:
            RuntimeError: 摄像头打开失败
        """
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (索引: {camera_index})")
        
        # 使用常量而不是魔数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        logger.info(f"摄像头已打开 (分辨率: {width}x{height})")
    
    def read_frame(self) -> Tuple[bool, Optional]:
        """读取一帧
        
        Returns:
            (成功标志, 帧数据)
        """
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            logger.info("摄像头已释放")


class FrameDrawer:
    """帧绘图器"""
    
    def __init__(self, config: dict):
        self.config = config
    
    def draw_detections(self, frame, detections: list):
        """在帧上绘制检测结果
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            
        Returns:
            绘制后的图像
        """
        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["coords"]
            
            # 绘制检测框
            cv2.rectangle(
                frame, 
                (x1, y1), 
                (x2, y2),
                self.config["box_color"],
                self.config["box_thickness"]
            )
            
            # 绘制标签和置信度
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


def main():
    """主函数"""
    
    detector = None
    camera = None
    drawer = None
    
    try:
        # 初始化
        detector = YOLODetector(CONFIG)
        camera = CameraCapture(
            CONFIG["camera_index"],
            CONFIG["frame_width"],
            CONFIG["frame_height"]
        )
        drawer = FrameDrawer(CONFIG)
        
        logger.info("系统初始化完成，开始检测（按 'q' 退出）")
        
        frame_count = 0
        last_detections = []
        
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                logger.warning("无法读取帧，摄像头可能已断开")
                break
            
            frame_count += 1
            
            # 只在指定间隔执行检测
            if frame_count % CONFIG["detection_interval"] == 0:
                results = detector.detect(frame)
                last_detections = detector.get_detections(results)
                logger.debug(f"帧 {frame_count}: 检测到 {len(last_detections)} 个目标")
            
            # 每帧都绘制（保持流畅）
            display_frame = drawer.draw_detections(frame.copy(), last_detections)
            
            # 显示FPS和帧数
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("YOLOv8 CPU Detection", display_frame)
            
            # 等待按键 (1ms超时)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("用户按下 'q'，正在退出...")
                break
        
        logger.info(f"程序正常退出，共处理 {frame_count} 帧")
    
    except RuntimeError as e:
        logger.error(f"运行时错误: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"未预期的错误: {e}", exc_info=True)
        return 1
    
    finally:
        # 确保资源释放
        if camera is not None:
            camera.release()
        
        cv2.destroyAllWindows()
        logger.info("资源已释放")
    
    return 0


if __name__ == "__main__":
    exit(main())