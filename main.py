import cv2
import numpy as np
import time
import os
import json
import argparse
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
                             QCheckBox, QProgressBar, QTextEdit, QGroupBox, QSpinBox,
                             QGridLayout, QRadioButton, QButtonGroup, QSlider, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont
import sys

# ====================== 全局配置 ======================
DEFAULT_CONFIG_PATH = "fish_config.json"
DEFAULT_MIN_AREA = 500
DEFAULT_TRACKER_TYPE = "CSRT"
# 视频预览最大尺寸（解决视频框变大问题）
MAX_PREVIEW_WIDTH = 800
MAX_PREVIEW_HEIGHT = 600


# ====================== 工具函数：配置文件管理 ======================
def create_default_config(config_path=DEFAULT_CONFIG_PATH):
    """创建默认配置文件"""
    default_config = {
        "fish_types": {
            "red_fish": {"h_lower": 0, "s_lower": 20, "v_lower": 30, "h_upper": 15, "s_upper": 255, "v_upper": 255},
            "blue_fish": {"h_lower": 100, "s_lower": 20, "v_lower": 20, "h_upper": 124, "s_upper": 255, "v_upper": 255},
            "green_fish": {"h_lower": 35, "s_lower": 25, "v_lower": 20, "h_upper": 77, "s_upper": 255, "v_upper": 255},
            "custom_fish": {"h_lower": 0, "s_lower": 30, "v_lower": 30, "h_upper": 30, "s_upper": 255, "v_upper": 255}
        },
        "min_contour_area": DEFAULT_MIN_AREA,
        "tracker_type": DEFAULT_TRACKER_TYPE,
        "dynamic_threshold": True,
        "save_video": True,
        "morphology_kernel": 5,  # 形态学操作核大小
        "contour_aspect_ratio": [0.2, 5.0],  # 轮廓宽高比范围（过滤异常形状）
        "multi_hsv": False,  # 启用多HSV阈值组合
        "multi_hsv_config": {
            "hsv1": {"h_lower": 0, "s_lower": 20, "v_lower": 20, "h_upper": 20, "s_upper": 255, "v_upper": 255},
            "hsv2": {"h_lower": 160, "s_lower": 20, "v_lower": 20, "h_upper": 179, "s_upper": 255, "v_upper": 255}
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=4)
    return default_config


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """加载配置文件，不存在则创建默认配置；存在则补充缺失的键"""
    default_config = create_default_config()  # 先获取默认配置

    if not os.path.exists(config_path):
        print(f"配置文件不存在，创建默认配置：{config_path}")
        return default_config

    # 读取现有配置
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            user_config = json.load(f)
        except json.JSONDecodeError:
            print("配置文件损坏，创建新的默认配置")
            return create_default_config(config_path)

    # 补充缺失的键（兼容性处理）
    updated_config = default_config.copy()
    for key in default_config.keys():
        if key in user_config:
            # 如果是字典，递归补充子键
            if isinstance(default_config[key], dict) and isinstance(user_config.get(key), dict):
                for sub_key in default_config[key].keys():
                    if sub_key in user_config[key]:
                        updated_config[key][sub_key] = user_config[key][sub_key]
            else:
                updated_config[key] = user_config[key]

    # 保存更新后的配置（补充缺失键）
    save_config(updated_config, config_path)
    return updated_config


def save_config(config, config_path=DEFAULT_CONFIG_PATH):
    """保存配置到文件"""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


# ====================== 工具函数：动态阈值计算 ======================
def get_dynamic_v_threshold(frame, base_v_lower=20, v_offset=10):
    """根据帧亮度动态调整V值下限"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness_avg = np.mean(gray)

    if brightness_avg < 50:  # 暗帧
        dynamic_v = base_v_lower - v_offset
    elif brightness_avg > 200:  # 亮帧
        dynamic_v = base_v_lower + v_offset
    else:
        dynamic_v = base_v_lower

    return max(0, min(dynamic_v, 255))


def get_fish_hsv_by_clustering(frame, roi=None, n_clusters=2):
    """通过K-Means聚类自动提取HSV阈值"""
    if roi is None:
        roi = (0, 0, frame.shape[1], frame.shape[0])
    x, y, w, h = roi
    roi_frame = frame[y:y + h, x:x + w]

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    hsv_reshaped = hsv.reshape(-1, 3)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(hsv_reshaped)
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        fish_cluster_idx = np.argsort(cluster_sizes)[-2]  # 次大聚类为鱼类
        fish_hsv = centers[fish_cluster_idx]

        lower_hsv = np.array([
            max(0, fish_hsv[0] - 10),
            max(0, fish_hsv[1] - 50),
            max(0, fish_hsv[2] - 50)
        ])
        upper_hsv = np.array([
            min(179, fish_hsv[0] + 10),
            min(255, fish_hsv[1] + 50),
            min(255, fish_hsv[2] + 50)
        ])
        return lower_hsv, upper_hsv
    except Exception as e:
        print(f"聚类提取阈值失败：{e}")
        return np.array([0, 30, 30]), np.array([30, 255, 255])


# ====================== 工具函数：HSV调优工具（独立窗口） ======================
def hsv_tuner_tool(video_path, config_path=DEFAULT_CONFIG_PATH):
    """HSV阈值调优工具，调整后保存到配置文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return

    # 创建窗口和滑动条
    cv2.namedWindow("HSV Tuner (S保存/Q退出)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tuner (S保存/Q退出)", 1000, 600)
    cv2.createTrackbar("H Lower", "HSV Tuner (S保存/Q退出)", 0, 179, lambda x: x)
    cv2.createTrackbar("H Upper", "HSV Tuner (S保存/Q退出)", 30, 179, lambda x: x)
    cv2.createTrackbar("S Lower", "HSV Tuner (S保存/Q退出)", 30, 255, lambda x: x)
    cv2.createTrackbar("S Upper", "HSV Tuner (S保存/Q退出)", 255, 255, lambda x: x)
    cv2.createTrackbar("V Lower", "HSV Tuner (S保存/Q退出)", 30, 255, lambda x: x)
    cv2.createTrackbar("V Upper", "HSV Tuner (S保存/Q退出)", 255, 255, lambda x: x)
    cv2.createTrackbar("Kernel Size", "HSV Tuner (S保存/Q退出)", 5, 15, lambda x: x if x % 2 == 1 else x + 1)

    # 加载当前配置
    config = load_config(config_path)
    custom_fish = config["fish_types"]["custom_fish"]
    cv2.setTrackbarPos("H Lower", "HSV Tuner (S保存/Q退出)", custom_fish["h_lower"])
    cv2.setTrackbarPos("H Upper", "HSV Tuner (S保存/Q退出)", custom_fish["h_upper"])
    cv2.setTrackbarPos("S Lower", "HSV Tuner (S保存/Q退出)", custom_fish["s_lower"])
    cv2.setTrackbarPos("S Upper", "HSV Tuner (S保存/Q退出)", custom_fish["s_upper"])
    cv2.setTrackbarPos("V Lower", "HSV Tuner (S保存/Q退出)", custom_fish["v_lower"])
    cv2.setTrackbarPos("V Upper", "HSV Tuner (S保存/Q退出)", custom_fish["v_upper"])
    cv2.setTrackbarPos("Kernel Size", "HSV Tuner (S保存/Q退出)", config["morphology_kernel"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
            ret, frame = cap.read()

        # 获取滑动条值
        h_lower = cv2.getTrackbarPos("H Lower", "HSV Tuner (S保存/Q退出)")
        h_upper = cv2.getTrackbarPos("H Upper", "HSV Tuner (S保存/Q退出)")
        s_lower = cv2.getTrackbarPos("S Lower", "HSV Tuner (S保存/Q退出)")
        s_upper = cv2.getTrackbarPos("S Upper", "HSV Tuner (S保存/Q退出)")
        v_lower = cv2.getTrackbarPos("V Lower", "HSV Tuner (S保存/Q退出)")
        v_upper = cv2.getTrackbarPos("V Upper", "HSV Tuner (S保存/Q退出)")
        kernel_size = cv2.getTrackbarPos("Kernel Size", "HSV Tuner (S保存/Q退出)")
        kernel_size = kernel_size if kernel_size % 2 == 1 else 5

        # 生成掩码和结果
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([h_lower, s_lower, v_lower]), np.array([h_upper, s_upper, v_upper]))

        # 形态学操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        # 显示
        cv2.imshow("Original", cv2.resize(frame, (640, 480)))
        combined = np.hstack([cv2.resize(mask, (500, 300)), cv2.resize(result, (500, 300))])
        cv2.imshow("HSV Tuner (S保存/Q退出)", combined)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存到配置文件
            config = load_config(config_path)
            config["fish_types"]["custom_fish"] = {
                "h_lower": h_lower, "s_lower": s_lower, "v_lower": v_lower,
                "h_upper": h_upper, "s_upper": s_upper, "v_upper": v_upper
            }
            config["morphology_kernel"] = kernel_size
            save_config(config, config_path)
            print(f"阈值已保存到配置文件：{config_path}")
            cv2.putText(result, "Saved!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            combined = np.hstack([cv2.resize(mask, (500, 300)), cv2.resize(result, (500, 300))])
            cv2.imshow("HSV Tuner (S保存/Q退出)", combined)
            cv2.waitKey(500)

    cap.release()
    cv2.destroyAllWindows()


# ====================== 追踪器初始化 ======================
def init_tracker(tracker_type):
    try:
        if tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        else:
            raise ValueError(f"不支持的追踪器类型: {tracker_type}")
    except AttributeError:
        if tracker_type == 'CSRT':
            tracker = cv2.legacy.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            tracker = cv2.legacy.TrackerKCF_create()
        else:
            raise ValueError(f"不支持的追踪器类型: {tracker_type}")
    return tracker


# ====================== 鱼类检测函数（优化识别率） ======================
def detect_fish(frame, min_area, lower_hsv, upper_hsv, config):
    """
    优化后的鱼类检测函数：
    1. 支持动态V值调整
    2. 支持多HSV阈值组合
    3. 增加轮廓宽高比筛选
    4. 可配置形态学操作核大小
    """
    dynamic_threshold = config["dynamic_threshold"]
    multi_hsv = config["multi_hsv"]
    multi_hsv_config = config["multi_hsv_config"]
    kernel_size = config["morphology_kernel"]
    aspect_ratio_min, aspect_ratio_max = config["contour_aspect_ratio"]

    # 1. 动态调整V值下限
    if dynamic_threshold:
        dynamic_v = get_dynamic_v_threshold(frame, base_v_lower=lower_hsv[2])
        lower_hsv[2] = dynamic_v

    # 2. 转换色彩空间并去噪
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)

    # 3. 多HSV阈值组合（如红色分两段）
    if multi_hsv:
        mask1 = cv2.inRange(blur_frame,
                            np.array([multi_hsv_config["hsv1"]["h_lower"], multi_hsv_config["hsv1"]["s_lower"],
                                      multi_hsv_config["hsv1"]["v_lower"]]),
                            np.array([multi_hsv_config["hsv1"]["h_upper"], multi_hsv_config["hsv1"]["s_upper"],
                                      multi_hsv_config["hsv1"]["v_upper"]]))
        mask2 = cv2.inRange(blur_frame,
                            np.array([multi_hsv_config["hsv2"]["h_lower"], multi_hsv_config["hsv2"]["s_lower"],
                                      multi_hsv_config["hsv2"]["v_lower"]]),
                            np.array([multi_hsv_config["hsv2"]["h_upper"], multi_hsv_config["hsv2"]["s_upper"],
                                      multi_hsv_config["hsv2"]["v_upper"]]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(blur_frame, lower_hsv, upper_hsv)

    # 4. 形态学操作（可配置核大小）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # 闭运算：先膨胀后腐蚀，填充小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. 查找轮廓并筛选
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fish_bboxes = []
    for contour in contours:
        # 筛选面积
        if cv2.contourArea(contour) < min_area:
            continue
        # 筛选宽高比（过滤异常形状）
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h != 0 else 0
        if not (aspect_ratio_min <= aspect_ratio <= aspect_ratio_max):
            continue
        # 筛选轮廓圆形度（可选，鱼类轮廓更接近圆形/椭圆形）
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)
        if circularity < 0.1:  # 过滤过于不规则的轮廓
            continue
        fish_bboxes.append([x, y, w, h])

    return fish_bboxes


# ====================== 追踪线程类 ======================
class TrackThread(QThread):
    update_frame = pyqtSignal(np.ndarray)
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    update_stats = pyqtSignal(int, float, float)
    finished_signal = pyqtSignal()

    def __init__(self, video_path, fish_type, config, auto_cluster=False):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self.fish_type = fish_type
        self.auto_cluster = auto_cluster
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            if not os.path.exists(self.video_path):
                self.update_log.emit(f"错误：视频文件不存在 - {self.video_path}")
                return

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.update_log.emit(f"错误：无法打开视频文件 - {self.video_path}")
                return

            # 视频参数
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.update_log.emit(f"视频信息：{video_width}x{video_height} | FPS：{video_fps:.2f} | 总帧数：{total_frames}")

            # 加载阈值
            fish_config = self.config["fish_types"][self.fish_type]
            lower_hsv = np.array([fish_config["h_lower"], fish_config["s_lower"], fish_config["v_lower"]])
            upper_hsv = np.array([fish_config["h_upper"], fish_config["s_upper"], fish_config["v_upper"]])
            min_area = self.config["min_contour_area"]
            tracker_type = self.config["tracker_type"]
            save_video = self.config["save_video"]

            # 自动聚类提取阈值
            if self.auto_cluster:
                ret, first_frame = cap.read()
                if ret:
                    self.update_log.emit("正在通过聚类自动提取鱼类HSV阈值...")
                    lower_hsv, upper_hsv = get_fish_hsv_by_clustering(first_frame)
                    self.update_log.emit(f"自动提取阈值：lower={lower_hsv}, upper={upper_hsv}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置帧位置

            # 视频保存
            video_writer = None
            if save_video:
                save_dir = 'runs/underwater_fish_track'
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"fish_track_{self.fish_type}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(save_path, fourcc, video_fps, (video_width, video_height))
                self.update_log.emit(f"结果视频将保存至：{save_path}")

            # 追踪器初始化
            trackers = {}
            fish_ids = set()
            next_track_id = 1
            processed_frames = 0
            start_time = time.time()

            # 逐帧处理
            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                display_frame = frame.copy()

                # 每10帧重新检测（可配置检测间隔）
                if processed_frames % 10 == 0:
                    fish_bboxes = detect_fish(display_frame, min_area, lower_hsv.copy(), upper_hsv.copy(), self.config)
                    trackers.clear()
                    fish_ids.clear()
                    next_track_id = 1
                    for bbox in fish_bboxes:
                        tracker = init_tracker(tracker_type)
                        # 追踪器初始化容错：防止无效bbox
                        if bbox[2] > 0 and bbox[3] > 0:
                            tracker.init(display_frame, tuple(bbox))
                            trackers[next_track_id] = tracker
                            fish_ids.add(next_track_id)
                            next_track_id += 1
                else:
                    # 更新追踪器（增加稳定性判断）
                    for track_id in list(trackers.keys()):
                        success, bbox = trackers[track_id].update(display_frame)
                        # 过滤异常bbox（宽高为0或超出帧范围）
                        if not success or bbox[2] <= 0 or bbox[3] <= 0:
                            del trackers[track_id]
                            fish_ids.discard(track_id)
                        else:
                            x, y, w, h = map(int, bbox)
                            # 确保bbox在帧范围内
                            x = max(0, min(x, video_width - 1))
                            y = max(0, min(y, video_height - 1))
                            w = max(1, min(w, video_width - x))
                            h = max(1, min(h, video_height - y))
                            # 绘制追踪框
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(display_frame, f'Fish ID: {track_id}', (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 统计信息
                processed_frames += 1
                progress = int((processed_frames / total_frames) * 100)
                elapsed_time = time.time() - start_time
                avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0

                # 发送信号（缩放到预览尺寸，解决视频框变大问题）
                preview_frame = cv2.resize(display_frame,
                                           (min(MAX_PREVIEW_WIDTH, video_width), min(MAX_PREVIEW_HEIGHT, video_height)),
                                           interpolation=cv2.INTER_AREA)
                self.update_frame.emit(preview_frame)
                self.update_progress.emit(progress)
                self.update_stats.emit(len(fish_ids), avg_fps, progress)
                self.update_log.emit(f"进度：{progress}% | 鱼类数：{len(fish_ids)} | FPS：{avg_fps:.1f}")

                # 保存视频（原尺寸）
                if save_video and video_writer is not None:
                    video_writer.write(display_frame)

                cv2.waitKey(1)

            # 资源释放
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()

            # 最终统计
            total_time = time.time() - start_time
            self.update_log.emit(
                f"\n追踪完成！总耗时：{total_time:.2f}秒 | 平均FPS：{avg_fps:.1f} | 检测鱼类数：{len(fish_ids)}")
            self.finished_signal.emit()

        except Exception as e:
            self.update_log.emit(f"程序出错：{str(e)}")
            self.finished_signal.emit()


# ====================== 主窗口类 ======================
class FishTrackUI(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.track_thread = None
        self.setWindowTitle("水下鱼类识别追踪系统（优化版）")
        self.setGeometry(100, 100, 1400, 800)
        self.init_ui()

    def init_ui(self):
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：参数配置区
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setMinimumWidth(450)

        # 新增：重置配置按钮
        reset_btn = QPushButton("重置配置为默认值")
        reset_btn.clicked.connect(self.reset_config)
        left_layout.addWidget(reset_btn)

        # 1. 视频选择
        video_group = QGroupBox("视频选择")
        video_layout = QVBoxLayout(video_group)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("请选择视频文件...")
        video_btn = QPushButton("选择视频")
        video_btn.clicked.connect(self.select_video)
        self.tuner_btn = QPushButton("打开HSV调优工具")
        self.tuner_btn.clicked.connect(self.open_hsv_tuner)
        video_layout.addWidget(self.video_path_edit)
        video_layout.addWidget(video_btn)
        video_layout.addWidget(self.tuner_btn)
        left_layout.addWidget(video_group)

        # 2. 鱼类类型选择与HSV配置
        fish_group = QGroupBox("鱼类类型与阈值配置")
        fish_layout = QGridLayout(fish_group)

        # 鱼类类型单选框
        self.fish_type_group = QButtonGroup()
        fish_types = list(self.config["fish_types"].keys())
        for i, fish_type in enumerate(fish_types):
            radio_btn = QRadioButton(fish_type)
            self.fish_type_group.addButton(radio_btn, i)
            fish_layout.addWidget(radio_btn, i, 0)
            if fish_type == "custom_fish":
                radio_btn.setChecked(True)

        # HSV参数显示与编辑
        hsv_labels = ["H下限", "S下限", "V下限", "H上限", "S上限", "V上限"]
        self.hsv_spins = {}
        for i, label in enumerate(hsv_labels):
            fish_layout.addWidget(QLabel(label), 0, i + 1)
            spin = QSpinBox()
            spin.setRange(0, 179 if "H" in label else 255)
            self.hsv_spins[label] = spin
            fish_layout.addWidget(spin, 1, i + 1)

        # 加载当前选中鱼类的阈值
        self.load_fish_hsv("custom_fish")
        self.fish_type_group.buttonClicked.connect(self.on_fish_type_change)

        # 高级配置：多HSV阈值
        self.multi_hsv_check = QCheckBox("启用多HSV阈值组合（如红色分两段）")
        self.multi_hsv_check.setChecked(self.config["multi_hsv"])
        fish_layout.addWidget(self.multi_hsv_check, len(fish_types) + 1, 0, 1, 7)

        # 自动聚类按钮
        self.auto_cluster_btn = QPushButton("自动提取阈值（聚类）")
        self.auto_cluster_btn.clicked.connect(self.auto_extract_hsv)
        fish_layout.addWidget(self.auto_cluster_btn, len(fish_types) + 2, 0, 1, 7)
        left_layout.addWidget(fish_group)

        # 3. 高级检测参数
        adv_group = QGroupBox("高级检测参数")
        adv_layout = QGridLayout(adv_group)

        # 最小轮廓面积
        adv_layout.addWidget(QLabel("最小轮廓面积:"), 0, 0)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 10000)
        self.min_area_spin.setValue(self.config["min_contour_area"])
        adv_layout.addWidget(self.min_area_spin, 0, 1)

        # 形态学核大小
        adv_layout.addWidget(QLabel("形态学核大小:"), 1, 0)
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(3, 15)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(self.config["morphology_kernel"])
        adv_layout.addWidget(self.kernel_spin, 1, 1)

        # 轮廓宽高比
        adv_layout.addWidget(QLabel("最小宽高比(×0.1):"), 2, 0)
        self.aspect_min = QSlider(Qt.Horizontal)
        self.aspect_min.setRange(1, 20)
        self.aspect_min.setValue(int(self.config["contour_aspect_ratio"][0] * 10))
        adv_layout.addWidget(self.aspect_min, 2, 1)
        adv_layout.addWidget(QLabel("最大宽高比:"), 3, 0)
        self.aspect_max = QSlider(Qt.Horizontal)
        self.aspect_max.setRange(1, 50)
        self.aspect_max.setValue(int(self.config["contour_aspect_ratio"][1]))
        adv_layout.addWidget(self.aspect_max, 3, 1)

        # 动态阈值
        self.dynamic_check = QCheckBox("启用动态V值阈值（适应光线）")
        self.dynamic_check.setChecked(self.config["dynamic_threshold"])
        adv_layout.addWidget(self.dynamic_check, 4, 0, 1, 2)

        left_layout.addWidget(adv_group)

        # 4. 追踪参数
        track_group = QGroupBox("追踪参数")
        track_layout = QGridLayout(track_group)

        # 追踪器类型
        track_layout.addWidget(QLabel("追踪器类型:"), 0, 0)
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["CSRT（精准）", "KCF（快速）"])
        self.tracker_combo.setCurrentText("CSRT（精准）" if self.config["tracker_type"] == "CSRT" else "KCF（快速）")
        track_layout.addWidget(self.tracker_combo, 0, 1)

        # 保存视频
        self.save_check = QCheckBox("保存结果视频")
        self.save_check.setChecked(self.config["save_video"])
        track_layout.addWidget(self.save_check, 1, 0, 1, 2)
        left_layout.addWidget(track_group)

        # 5. 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始追踪")
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn = QPushButton("停止追踪")
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.stop_btn.setEnabled(False)
        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_current_config)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.save_config_btn)
        left_layout.addLayout(btn_layout)

        # 6. 进度与统计
        progress_group = QGroupBox("进度与统计")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.fish_count_label = QLabel("当前检测鱼类数：0")
        self.fps_label = QLabel("平均FPS：0.0")
        self.progress_label = QLabel("进度：0%")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.fish_count_label)
        progress_layout.addWidget(self.fps_label)
        progress_layout.addWidget(self.progress_label)
        left_layout.addWidget(progress_group)

        # 7. 日志输出
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        # 右侧：视频显示区（固定最大尺寸）
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.video_label = QLabel("视频预览区")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid gray;")
        # 固定视频预览区尺寸
        self.video_label.setMinimumSize(MAX_PREVIEW_WIDTH, MAX_PREVIEW_HEIGHT)
        self.video_label.setMaximumSize(MAX_PREVIEW_WIDTH, MAX_PREVIEW_HEIGHT)
        right_layout.addWidget(self.video_label)

        # 添加到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget, stretch=1)

    def reset_config(self):
        """重置配置为默认值"""
        reply = QMessageBox.question(self, "重置配置", "确定要重置所有配置为默认值吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            create_default_config()
            self.config = load_config()
            self.load_fish_hsv("custom_fish")
            self.min_area_spin.setValue(self.config["min_contour_area"])
            self.kernel_spin.setValue(self.config["morphology_kernel"])
            self.aspect_min.setValue(int(self.config["contour_aspect_ratio"][0] * 10))
            self.aspect_max.setValue(int(self.config["contour_aspect_ratio"][1]))
            self.dynamic_check.setChecked(self.config["dynamic_threshold"])
            self.multi_hsv_check.setChecked(self.config["multi_hsv"])
            self.tracker_combo.setCurrentText("CSRT（精准）" if self.config["tracker_type"] == "CSRT" else "KCF（快速）")
            self.save_check.setChecked(self.config["save_video"])
            self.log_text.append("配置已重置为默认值！")

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.video_path_edit.setText(file_path)

    def open_hsv_tuner(self):
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            self.log_text.append("请先选择视频文件！")
            return
        hsv_tuner_tool(video_path)
        # 重新加载配置
        self.config = load_config()
        self.load_fish_hsv("custom_fish")

    def load_fish_hsv(self, fish_type):
        """加载指定鱼类的HSV阈值到控件"""
        fish_config = self.config["fish_types"][fish_type]
        self.hsv_spins["H下限"].setValue(fish_config["h_lower"])
        self.hsv_spins["S下限"].setValue(fish_config["s_lower"])
        self.hsv_spins["V下限"].setValue(fish_config["v_lower"])
        self.hsv_spins["H上限"].setValue(fish_config["h_upper"])
        self.hsv_spins["S上限"].setValue(fish_config["s_upper"])
        self.hsv_spins["V上限"].setValue(fish_config["v_upper"])

    def on_fish_type_change(self, radio_btn):
        """切换鱼类类型时更新HSV控件"""
        fish_type = radio_btn.text()
        self.load_fish_hsv(fish_type)

    def auto_extract_hsv(self):
        """自动提取阈值并更新控件"""
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            self.log_text.append("请先选择视频文件！")
            return

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.log_text.append("无法读取视频帧！")
            return

        self.log_text.append("正在通过聚类提取阈值...")
        lower_hsv, upper_hsv = get_fish_hsv_by_clustering(frame)
        self.hsv_spins["H下限"].setValue(int(lower_hsv[0]))
        self.hsv_spins["S下限"].setValue(int(lower_hsv[1]))
        self.hsv_spins["V下限"].setValue(int(lower_hsv[2]))
        self.hsv_spins["H上限"].setValue(int(upper_hsv[0]))
        self.hsv_spins["S上限"].setValue(int(upper_hsv[1]))
        self.hsv_spins["V上限"].setValue(int(upper_hsv[2]))
        self.log_text.append(f"自动提取阈值完成：lower={lower_hsv}, upper={upper_hsv}")

    def save_current_config(self):
        """保存当前控件参数到配置文件"""
        selected_fish = self.fish_type_group.checkedButton().text()
        # 更新基础HSV配置
        self.config["fish_types"][selected_fish] = {
            "h_lower": self.hsv_spins["H下限"].value(),
            "s_lower": self.hsv_spins["S下限"].value(),
            "v_lower": self.hsv_spins["V下限"].value(),
            "h_upper": self.hsv_spins["H上限"].value(),
            "s_upper": self.hsv_spins["S上限"].value(),
            "v_upper": self.hsv_spins["V上限"].value()
        }
        # 更新高级参数
        self.config["min_contour_area"] = self.min_area_spin.value()
        self.config["morphology_kernel"] = self.kernel_spin.value()
        self.config["contour_aspect_ratio"] = [self.aspect_min.value() / 10, self.aspect_max.value()]
        self.config["dynamic_threshold"] = self.dynamic_check.isChecked()
        self.config["multi_hsv"] = self.multi_hsv_check.isChecked()
        self.config["tracker_type"] = "CSRT" if self.tracker_combo.currentText().startswith("CSRT") else "KCF"
        self.config["save_video"] = self.save_check.isChecked()
        # 保存配置
        save_config(self.config)
        self.log_text.append("配置已保存到文件！")

    def start_tracking(self):
        """开始追踪"""
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            self.log_text.append("请先选择视频文件！")
            return

        # 保存当前配置
        self.save_current_config()
        selected_fish = self.fish_type_group.checkedButton().text()

        # 启动追踪线程
        self.track_thread = TrackThread(video_path, selected_fish, self.config, self.auto_cluster_btn.isChecked())
        self.track_thread.update_frame.connect(self.show_frame)
        self.track_thread.update_log.connect(self.append_log)
        self.track_thread.update_progress.connect(self.update_progress)
        self.track_thread.update_stats.connect(self.update_stats)
        self.track_thread.finished_signal.connect(self.track_finished)

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.track_thread.start()

    def stop_tracking(self):
        """停止追踪"""
        if self.track_thread and self.track_thread.isRunning():
            self.track_thread.stop()
            self.append_log("正在停止追踪...")
            self.stop_btn.setEnabled(False)

    def show_frame(self, frame):
        """显示视频帧（固定尺寸，解决变大问题）"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 保持比例填充到固定尺寸
        pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.KeepAspectRatio,
                                                    Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def append_log(self, text):
        """添加日志"""
        self.log_text.append(text)
        self.log_text.moveCursor(self.log_text.textCursor().End)

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"进度：{value}%")

    def update_stats(self, fish_count, fps, progress):
        """更新统计信息"""
        self.fish_count_label.setText(f"当前检测鱼类数：{fish_count}")
        self.fps_label.setText(f"平均FPS：{fps:.1f}")
        self.progress_label.setText(f"进度：{progress:.1f}%")

    def track_finished(self):
        """追踪完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("追踪线程已结束")


# ====================== 命令行参数解析与程序入口 ======================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="水下鱼类识别追踪系统（优化版）")
    parser.add_argument('--source', type=str, help="视频路径（可选，也可在UI中选择）")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument('--fish-type', type=str, default="custom_fish",
                        help="鱼类类型（red_fish/blue_fish/green_fish/custom_fish）")
    parser.add_argument('--tuner', action='store_true', help="直接打开HSV调优工具")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 直接打开调优工具
    if args.tuner:
        if not args.source:
            print("使用--tuner时需指定--source视频路径")
            return
        hsv_tuner_tool(args.source, args.config)
        return

    # 启动UI
    app = QApplication(sys.argv)
    window = FishTrackUI(config)
    # 如果命令行指定了视频路径，自动填入
    if args.source:
        window.video_path_edit.setText(args.source)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()