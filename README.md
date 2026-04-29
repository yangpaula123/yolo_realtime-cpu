# YOLOv8n 实时目标检测 Web 应用

基于Streamlit的YOLOv8n实时目标检测Web应用，支持通过浏览器进行实时视频流检测。

## 功能特点

- 实时目标检测
- 摄像头视频流处理
- 可调节的检测参数
- 检测结果实时统计
- 可自定义检测框颜色
- 多类别目标检测

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

### Streamlit Web应用

```bash
streamlit run streamlit_app.py
```

运行后，浏览器会自动打开应用页面。

### 原始OpenCV应用

```bash
python app.py
```

## 配置说明

主要配置参数在`config.py`中：

- `model_name`: 模型文件名（默认：yolov8n.pt）
- `frame_width`: 视频帧宽度（默认：640）
- `frame_height`: 视频帧高度（默认：480）
- `detection_interval`: 检测间隔帧数（默认：2）
- `target_classes`: 目标类别列表（默认：["person", "cell phone", "book"]）
- `conf_threshold`: 置信度阈值（默认：0.3）
- `camera_index`: 摄像头索引（默认：0）

## 使用说明

1. 启动应用后，左侧面板可以调整检测参数：
   - 置信度阈值：控制检测的严格程度
   - 检测间隔：控制检测频率（帧数）
   - 目标类别：选择要检测的目标类型
   - 边界框颜色：自定义检测框颜色

2. 右侧面板显示：
   - 检测统计：各类目标的检测数量
   - 检测结果：详细的检测结果列表

3. 点击"停止检测"按钮可以停止视频流处理

## 注意事项

- 确保摄像头未被其他程序占用
- 首次运行会自动下载YOLOv8n模型
- 建议使用Chrome或Edge浏览器以获得最佳性能
- 如果检测速度较慢，可以增加检测间隔或降低帧分辨率

## 技术栈

- Streamlit: Web应用框架
- YOLOv8n: 目标检测模型
- OpenCV: 视频处理
- NumPy: 数值计算
