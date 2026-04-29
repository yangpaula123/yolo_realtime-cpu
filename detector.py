from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import Optional
import logging


logger = logging.getLogger(__name__)

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