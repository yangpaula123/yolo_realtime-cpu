
import streamlit as st
import cv2
import numpy as np
import time
from detector import YOLODetector
from config import CONFIG
from drawer import FrameDrawer

# 设置页面配置
st.set_page_config(
    page_title="YOLOv8 实时目标检测",
    page_icon="🎯",
    layout="wide"
)

# 页面标题
st.title("🎯 YOLOv8 实时目标检测 Web 应用")
st.markdown("---")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 检测设置")

    # 置信度阈值
    conf_threshold = st.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=CONFIG["conf_threshold"],
        step=0.05
    )

    # 检测间隔
    detection_interval = st.slider(
        "检测间隔（帧）",
        min_value=1,
        max_value=10,
        value=CONFIG["detection_interval"],
        step=1
    )

    # 目标类别选择
    st.subheader("目标类别")
    available_classes = ["person", "cell phone", "book", "laptop", "bottle", "cup", 
                        "chair", "table", "tv", "mouse", "keyboard"]
    target_classes = st.multiselect(
        "选择要检测的类别",
        available_classes,
        default=CONFIG["target_classes"]
    )

    # 边界框颜色
    box_color = st.color_picker("边界框颜色", "#00FF00")

    # 更新配置
    CONFIG["conf_threshold"] = conf_threshold
    CONFIG["detection_interval"] = detection_interval
    CONFIG["target_classes"] = target_classes if target_classes else available_classes
    CONFIG["box_color"] = tuple(int(box_color[i:i+2], 16) for i in (1, 3, 5))

# 创建两列布局
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 实时检测画面")
    # 创建视频占位符
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 检测统计")
    # 创建统计信息占位符
    stats_placeholder = st.empty()

    st.subheader("📝 检测结果")
    # 创建检测结果占位符
    results_placeholder = st.empty()

# 初始化检测器和绘图器
@st.cache_resource
def init_detector():
    return YOLODetector(CONFIG)

@st.cache_resource
def init_drawer():
    return FrameDrawer(CONFIG)

detector = init_detector()
drawer = init_drawer()

# 打开摄像头
cap = cv2.VideoCapture(CONFIG["camera_index"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

# 检查摄像头是否成功打开
if not cap.isOpened():
    st.error("无法打开摄像头，请检查摄像头是否被占用或索引是否正确")
    st.stop()

# 创建停止按钮
stop_button = st.button("停止检测")

frame_count = 0
last_detections = []

# 主循环
while not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("无法从摄像头读取帧")
        break

    frame_count += 1

    # 执行检测
    if frame_count % CONFIG["detection_interval"] == 0:
        results = detector.detect(frame)
        last_detections = detector.get_detections(results)

    # 绘制检测结果
    display_frame = drawer.draw_detections(frame.copy(), last_detections)

    # 将BGR转换为RGB
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    # 显示视频帧
    video_placeholder.image(display_frame, channels="RGB", use_column_width=True)

    # 统计检测结果
    detection_stats = {}
    for det in last_detections:
        label = det["label"]
        detection_stats[label] = detection_stats.get(label, 0) + 1

    # 显示统计信息
    if detection_stats:
        stats_text = "\n".join([f"{label}: {count}" for label, count in detection_stats.items()])
        stats_placeholder.write(stats_text)
    else:
        stats_placeholder.write("暂无检测结果")

    # 显示详细检测结果
    if last_detections:
        results_text = "\n".join([
            f"• {det['label']}: {det['confidence']:.2f}"
            for det in last_detections
        ])
        results_placeholder.text(results_text)
    else:
        results_placeholder.text("暂无检测结果")

# 释放资源
cap.release()
