# RK3588 YOLO 实时缺陷检测

import cv2
import numpy as np
from rknn.api import RKNN
import os
import time
import psutil
import threading
from datetime import datetime
import json
import csv  # 添加CSV模块导入

RKNN_MODEL = r''
QUANTIZE_ON = True

OBJ_THRESH = 0.3
NMS_THRESH = 0.45
# 修改输入尺寸以匹配模型要求
IMG_SIZE = 640  

CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # 使用更快的插值方法
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_NEAREST)  # 改为INTER_NEAREST更快
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)  # 修正这里，将dwdh改为dh

def process_output(output, r, dwdh):
    num_classes = len(CLASSES)
    outputs = np.transpose(output.reshape((num_classes + 4), -1))
    
    # 提前过滤低置信度的检测框，减少后续计算量
    confidence = outputs[:, 4:].max(axis=1)
    mask = confidence > OBJ_THRESH
    if not np.any(mask):
        return []
    
    filtered_outputs = outputs[mask]
    scores = filtered_outputs[:, 4:(4 + num_classes)]
    boxes = filtered_outputs[:, :4]
    
    max_score_indices = np.argmax(scores, axis=1)
    max_scores = scores[np.arange(len(scores)), max_score_indices]
    
    # 直接使用numpy操作而不是循环
    boxes = np.divide(boxes, r)
    boxes = boxes - np.array(dwdh * 2)
    
    boxes = np.concatenate([
        boxes[:, :2] - boxes[:, 2:] / 2,
        boxes[:, :2] + boxes[:, 2:] / 2
    ], axis=1)
    
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_scores.tolist(), OBJ_THRESH, NMS_THRESH)
    
    results = []
    for idx in indices:
        box = boxes[idx]
        score = max_scores[idx]
        cls_id = max_score_indices[idx]
        results.append({
            'box': box.astype(int).tolist(),
            'score': float(score),
            'class_id': int(cls_id),
            'class_name': CLASSES[cls_id]
        })
    
    return results

def draw_boxes(img, results):
    # 只绘制置信度较高的框
    HIGH_CONF_THRESH = 0.5  # 高置信度阈值
    
    # 创建结果的副本，避免修改原始数据
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # 限制显示的检测框数量，避免过多框导致显示混乱
    MAX_BOXES = 10
    sorted_results = sorted_results[:MAX_BOXES]
    
    for det in sorted_results:
        box = det['box']
        score = det['score']
        class_name = det['class_name']
        
        # 根据置信度调整线条粗细和颜色深浅
        thickness = 2 if score > HIGH_CONF_THRESH else 1
        color_intensity = min(255, int(score * 255))  # 置信度越高颜色越深
        color = (0, 0, color_intensity + 150)  # 确保即使低置信度也有基本可见度
        
        # 绘制检测框
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
        
        # 只对高置信度的检测结果显示标签
        if score > HIGH_CONF_THRESH:
            # 创建更好的文本背景
            text = f'{class_name} {score:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (box[0], box[1] - text_height - 5), 
                         (box[0] + text_width, box[1]), color, -1)
            # 使用白色文本，提高可读性
            cv2.putText(img, text, (box[0], box[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img

class ResourceMonitor:
    def __init__(self):
        self.running = False
        self.max_cpu_percent = 0
        self.max_memory_percent = 0
        self.max_memory_used = 0
        self.max_npu_usage = 0
        self.npu_core_usage = [0, 0, 0]  # 存储各NPU核心的使用率
        self.max_npu_core_usage = [0, 0, 0]  # 存储各NPU核心的最大使用率
        self.max_fps = 0
        self.avg_fps = 0
        self.fps_values = []  # 用于计算平均FPS
        self.monitoring_thread = None
        self.lock = threading.Lock()
        self.log_file = None
        self.csv_writer = None
        self.start_time = time.time()

    def get_npu_usage(self):
        # 尝试读取NPU使用率
        npu_paths = [
            "/sys/devices/platform/rknpu/load",
            "/sys/kernel/debug/rknpu/load"
        ]
        
        # 尝试读取各个NPU核心的使用率
        npu_core_paths = [
            ["/sys/devices/platform/rknpu/load0", "/sys/kernel/debug/rknpu/load0"],
            ["/sys/devices/platform/rknpu/load1", "/sys/kernel/debug/rknpu/load1"],
            ["/sys/devices/platform/rknpu/load2", "/sys/kernel/debug/rknpu/load2"]
        ]
        
        # 读取总体NPU使用率
        total_usage = 0
        for path in npu_paths:
            try:
                with open(path, "r") as f:
                    load = f.read().strip()
                total_usage = float(load.rstrip(','))
                break
            except (FileNotFoundError, PermissionError):
                continue
        
        # 读取各个核心的使用率
        core_values = [0, 0, 0]
        for i, paths in enumerate(npu_core_paths):
            for path in paths:
                try:
                    with open(path, "r") as f:
                        load = f.read().strip()
                    core_values[i] = float(load.rstrip(','))
                    break
                except (FileNotFoundError, PermissionError):
                    continue
        
        # 更新核心使用率
        self.npu_core_usage = core_values
        
        # 更新最大核心使用率
        for i in range(3):
            self.max_npu_core_usage[i] = max(self.max_npu_core_usage[i], core_values[i])
        
        # 减少调试信息，只在有显著变化时打印NPU使用率
        # if any(self.npu_core_usage):
        #     print(f"NPU核心使用率: Core0={self.npu_core_usage[0]:.1f}%, Core1={self.npu_core_usage[1]:.1f}%, Core2={self.npu_core_usage[2]:.1f}%")
        
        return total_usage
        
    def start_monitoring(self, log_to_csv=True):
        self.running = True
        
        # 如果需要记录到CSV
        if log_to_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 修改为Linux路径格式，避免Windows路径在Linux系统上的问题
            self.log_file = open(f'performance_log_{timestamp}.csv', 'w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            # 写入CSV头
            self.csv_writer.writerow([
                'Time(s)', 'CPU(%)', 'Memory(%)', 'Memory(MB)', 
                'NPU Total(%)', 'NPU Core0(%)', 'NPU Core1(%)', 'NPU Core2(%)',
                'FPS'
            ])
        
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self._print_results()
        
        # 关闭CSV文件
        if self.log_file:
            self.log_file.close()
            print(f"性能数据已保存到CSV文件")
            
    def _monitor_resources(self):
        # 降低监控频率，减少系统开销
        while self.running:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            # 内存使用情况
            memory = psutil.virtual_memory()
            # NPU使用率
            npu_usage = self.get_npu_usage()
            
            # 当前FPS
            current_fps = self.max_fps if not self.fps_values else self.fps_values[-1]
            
            # 记录到CSV
            if self.csv_writer:
                elapsed_time = time.time() - self.start_time
                self.csv_writer.writerow([
                    f"{elapsed_time:.1f}",
                    f"{cpu_percent:.1f}",
                    f"{memory.percent:.1f}",
                    f"{memory.used / (1024 * 1024):.1f}",
                    f"{npu_usage:.1f}",
                    f"{self.npu_core_usage[0]:.1f}",
                    f"{self.npu_core_usage[1]:.1f}",
                    f"{self.npu_core_usage[2]:.1f}",
                    f"{current_fps:.1f}"
                ])
                self.log_file.flush()  # 确保数据立即写入文件
            
            with self.lock:
                self.max_cpu_percent = max(self.max_cpu_percent, cpu_percent)
                self.max_memory_percent = max(self.max_memory_percent, memory.percent)
                self.max_memory_used = max(self.max_memory_used, memory.used)
                self.max_npu_usage = max(self.max_npu_usage, npu_usage)
            
            time.sleep(1.0)  # 每秒更新一次
            
    def update_fps(self, current_fps):
        with self.lock:
            self.max_fps = max(self.max_fps, current_fps)
            # 限制存储的FPS值数量，避免内存无限增长
            self.fps_values.append(current_fps)
            if len(self.fps_values) > 100:  # 只保留最近100个值
                self.fps_values.pop(0)
            
    def _print_results(self):
        if self.fps_values:
            self.avg_fps = sum(self.fps_values) / len(self.fps_values)
            
        print("\n最大资源使用情况:")
        print(f"CPU使用率: {self.max_cpu_percent:.1f}%")
        print(f"内存使用率: {self.max_memory_percent:.1f}%")
        print(f"内存使用量: {self.max_memory_used / (1024 * 1024):.1f} MB")
        print(f"NPU总体使用率: {self.max_npu_usage:.1f}%")
        print(f"NPU核心最大使用率: Core0={self.max_npu_core_usage[0]:.1f}%, Core1={self.max_npu_core_usage[1]:.1f}%, Core2={self.max_npu_core_usage[2]:.1f}%")
        print(f"最大FPS: {self.max_fps:.1f}")
        print(f"平均FPS: {self.avg_fps:.1f}")

def main():
    # 初始化资源监控器
    resource_monitor = ResourceMonitor()
    
    # 声明全局变量
    global IMG_SIZE, OBJ_THRESH
    
    # 添加控制变量
    show_fps = True
    show_boxes = True
    
    # 初始化RKNN模型
    rknn = RKNN(verbose=False)  # 将verbose设为False减少调试信息
    print('--> Load RKNN model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed!')
        return
    
    # 由于无法获取模型输入形状，我们将使用默认的IMG_SIZE
    print(f'--> 使用默认输入尺寸: {IMG_SIZE}x{IMG_SIZE}')
    
    print('--> Init runtime environment')
    # 修改core_mask参数，使用有效值
    # 7 (二进制111) 表示使用所有3个NPU核心
    ret = rknn.init_runtime(target='rk3588', device_id=None, core_mask=7)
    if ret != 0:
        print('Init runtime environment failed!')
        return

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
        
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 获取实际的分辨率
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头实际分辨率: {actual_width}x{actual_height}")

    print("实时检测已启动，按 'q' 键退出")
    window_name = "PCB Defect Detection"
    
    # 启动资源监控
    resource_monitor.start_monitoring()

    # 用于计算FPS
    frame_times = [] # 存储最近几帧的处理时间
    fps = 0
    SMOOTHING_WINDOW = 10 # 使用最近10帧的时间来计算平均FPS
    
    # 添加跳帧处理
    frame_count = 0
    FRAME_SKIP = 1  # 每隔1帧处理一次，可以根据需要调整
    
    # 添加图像缩放因子，可以减小处理的图像尺寸提高性能
    SCALE_FACTOR = 1.0  # 1.0表示原始尺寸，0.5表示缩小一半

    while True:
        start_time = time.time() # 记录帧处理开始时间

        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头帧")
            break
        
        # 可选的图像缩放，提高处理速度
        if SCALE_FACTOR != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            
        frame_count += 1
        process_this_frame = frame_count % (FRAME_SKIP + 1) == 0
        
        if process_this_frame:
            # 预处理
            img_resized, r, dwdh = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
            # 确保输入数据格式正确
            img_input = np.expand_dims(img_resized, 0).astype(np.uint8)  # 确保数据类型为uint8
            
            # 减少调试信息
            # print(f"Input shape: {img_input.shape}, dtype: {img_input.dtype}")
            
            # 推理
            outputs = rknn.inference(inputs=[img_input])
            
            # 后处理
            results = process_output(outputs[0], r, dwdh)
            
            # 保存当前结果用于跳帧时显示
            last_results = results
        else:
            # 使用上一帧的结果
            results = last_results if 'last_results' in locals() else []
        
        # 绘制检测框和FPS
        if show_boxes:
            frame = draw_boxes(frame, results)

        # 计算和显示FPS
        end_time = time.time() # 记录帧处理结束时间
        frame_time = end_time - start_time
        frame_times.append(frame_time)
        if len(frame_times) > SMOOTHING_WINDOW:
            frame_times.pop(0) # 保持窗口大小

        if len(frame_times) > 0:
            avg_frame_time = sum(frame_times) / len(frame_times)
            if avg_frame_time > 0:
                fps = 1.0 / avg_frame_time
                resource_monitor.update_fps(fps)  # 更新FPS记录

        # 在画面上显示当前FPS
        if show_fps:
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示结果
        cv2.imshow(window_name, frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):  # 切换FPS显示
            show_fps = not show_fps
        elif key == ord('b'):  # 切换检测框显示
            show_boxes = not show_boxes
        elif key == ord('+') or key == ord('='):  # 提高检测阈值
            OBJ_THRESH = min(0.9, OBJ_THRESH + 0.05)
            print(f"检测阈值: {OBJ_THRESH:.2f}")
        elif key == ord('-'):  # 降低检测阈值
            OBJ_THRESH = max(0.1, OBJ_THRESH - 0.05)
            print(f"检测阈值: {OBJ_THRESH:.2f}")
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 在主循环结束后停止资源监控
    resource_monitor.stop_monitoring()
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    rknn.release()
    print("检测已结束")

if __name__ == '__main__':
    main()
