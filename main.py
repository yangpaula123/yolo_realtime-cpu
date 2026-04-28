import cv2
import numpy as np
import logging
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import Tuple, Optional

# 配置日志
logging.basicConfig( #  配置日志记录的基本设置
    level=logging.INFO, #  设置日志级别为INFO，即记录INFO级别及以上的日志
    format='%(asctime)s - %(levelname)s - %(message)s' #  设置日志格式，包含时间、日志级别和消息内容
)
logger = logging.getLogger(__name__) #  创建一个日志记录器，使用当前模块的名称作为标识

#  配置参数（便于修改）
CONFIG = {
    "model_name": "yolov8n.pt",
    "frame_width": 640, #  设置视频帧的宽度为640像素
    "frame_height": 480, #  设置视频帧的高度为480像素
    "detection_interval": 2,  # 每N帧检测一次
    "target_classes": ["person", "cell phone", "book"],
    "box_color": (0, 255, 0),
    "box_thickness": 2,
    "text_color": (0, 255, 0),
    "text_scale": 0.6,
    "text_thickness": 2,
    "conf_threshold": 0.3, #  置信度阈值，用于过滤检测结果，低于此值的检测将被忽略
    "camera_index": 0, #  指定使用系统中的第一个摄像头（通常默认为0）
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
        self.target_class_names = set(config["target_classes"]) #  从配置中获取目标类别并转换为集合类型
        self._init_model() #  调用私有方法初始化模型
    
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
    
    def detect(self, frame) -> Optional[Results]:
        """执行目标检测
        
        Args:
            frame: 输入图像
            
        Returns:
            YOLO检测结果对象
        """
        if self.model is None:
            return None
        
        results = self.model(frame, verbose=False) #  使用模型进行目标检测，不显示详细输出
        return results
    
    def get_detections(self, results) -> list:
        """提取目标检测结果
        
        Args:
            results: YOLO检测结果
            
        Returns:
            (label, confidence, box_coords) 的列表
        """
        detections = [] #  初始化检测结果列表
        
        if results is None:
            return detections
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0]) #  获取类别ID
                conf = float(box.conf[0]) #  获取置信度
                label = self.model.names[cls_id] #  获取类别标签
                
                # 过滤低置信度
                if conf < self.config["conf_threshold"]:
                    continue
                
                # 只保留目标类别
                if label not in self.target_class_names:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #  获取边界框坐标
                detections.append({ #  将检测结果添加到列表中
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
        self.cap = cv2.VideoCapture(camera_index) #  创建VideoCapture对象，初始化摄像头
        
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (索引: {camera_index})")
        
        # 使用常量而不是魔数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) #  设置摄像头的宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) #  设置摄像头的高度
        
        logger.info(f"摄像头已打开 (分辨率: {width}x{height})")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧
        
        Returns:
            (成功标志, 帧数据)
        """
        ret, frame = self.cap.read() #  从摄像头捕获设备读取一帧图像
        return ret, frame #  返回读取状态和帧数据
    
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
                frame,  #  输入图像
                (x1, y1),  #  检测框左上角坐标
                (x2, y2), #  检测框右下角坐标
                self.config["box_color"],
                self.config["box_thickness"]
            )
            
            # 绘制标签和置信度
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame, #  输入图像
                text, #  要绘制的文本
                (x1, y1 - 10), #  文本位置（在检测框上方）
                cv2.FONT_HERSHEY_SIMPLEX, #  字体类型
                self.config["text_scale"],
                self.config["text_color"],
                self.config["text_thickness"]
            )
        
        return frame #  返回处理后的帧


def main():
    """主函数"""
    
    detector = None #  检测器对象，用于执行目标检测
    camera = None #  摄像头捕获对象，用于获取视频帧
    drawer = None #  帧绘制器对象，用于在帧上绘制检测结果
    
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
        
        frame_count = 0 #  初始化帧计数器，用于统计处理的帧数
        last_detections = [] #  初始化上一次检测结果列表，用于存储上一帧的检测结果
        
        while True: #  进入主循环，持续处理视频帧
            ret, frame = camera.read_frame() #  读取一帧图像，ret表示是否成功读取，frame为图像数据
            if not ret:
                logger.warning("无法读取帧，摄像头可能已断开")
                break
            
            frame_count += 1
            
            # 只在指定间隔执行检测
            if frame_count % CONFIG["detection_interval"] == 0:
                results = detector.detect(frame) #  使用检测器进行目标检测
                last_detections = detector.get_detections(results) #  获取检测结果
                logger.debug(f"帧 {frame_count}: 检测到 {len(last_detections)} 个目标")
            
            # 每帧都绘制（保持流畅）
            display_frame = drawer.draw_detections(frame.copy(), last_detections)
            
            # 显示FPS和帧数
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, 30), #  文本位置在左上角(10,30)
                cv2.FONT_HERSHEY_SIMPLEX, #  使用SIMPLEX字体
                0.7, #  字体大小为0.7
                (0, 255, 0), #  使用绿色(0,255,0)
                2 #  线条粗细为2
            )
            
            cv2.imshow("YOLOv8 CPU Detection", display_frame) #  显示处理后的视频帧             窗口名称为"YOLOv8 CPU Detection"
            
            # 等待按键 (1ms超时)
            key = cv2.waitKey(1) & 0xFF #  使用cv2.waitKey函数等待按键输入，参数1表示等待1毫秒             & 0xFF 用于确保只取按键值的低8位，防止不同平台差异
            if key == ord('q'): #  检查是否按下'q'键
                logger.info("用户按下 'q'，正在退出...") #  记录用户按'q'键退出程序的信息
                break
        
        logger.info(f"程序正常退出，共处理 {frame_count} 帧")
    
    except RuntimeError as e: #  捕获运行时错误
        logger.error(f"运行时错误: {e}")
        return 1
    
    except Exception as e: #  捕获其他未预期的异常
        logger.error(f"未预期的错误: {e}", exc_info=True)
        return 1
    
    finally:
        # 确保资源释放
        if camera is not None:
            camera.release()
        
        cv2.destroyAllWindows() #  关闭所有OpenCV创建的窗口
        logger.info("资源已释放")
    
    return 0


if __name__ == "__main__": #  程序入口点
    exit(main()) #  调用main函数并以退出状态码结束程序