# RK3588-YOLO-Defect-Detection

基于RK3588芯片和YOLOv8的PCB缺陷检测系统，针对Orange Pi 5 Plus等RK3588开发板优化。

## 项目概述

本项目实现了在RK3588平台上部署YOLOv8模型用于PCB电路板缺陷检测，支持以下缺陷类型：
- missing_hole (缺失孔)
- mouse_bite (鼠咬)
- open_circuit (开路)
- short (短路)
- spur (毛刺)
- spurious_copper (多余铜箔)

系统支持FP32和INT8量化模型，可用于实时检测、单张图片检测和测试集评估。

## 功能特性

- 支持多种模型格式：FP32浮点模型和INT8量化模型
- 多种检测模式：
  - 单张图片检测
  - 测试集批量检测与评估
  - 模型性能对比分析
- 完整的评估指标：
  - Precision, Recall, F1 Score
  - mAP@0.5, mAP@0.5:0.95
  - 平均推理时间和FPS
- 针对RK3588 NPU优化

## 环境要求

- 硬件：
  - RK3588芯片的开发板（如Orange Pi 5 Plus）
  - 设备ID: 8f611d4cc9a75d34（可根据实际设备修改）

- 软件：
  - Python 3.6+
  - OpenCV
  - NumPy
  - RKNN Python API
  - tqdm

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/RK3588-YOLO-Defect-Detection.git
cd RK3588-YOLO-Defect-Detection
```

2. 安装依赖：
```bash
pip install opencv-python numpy tqdm
```

3. 安装RKNN工具包（根据您的RK3588设备选择合适的版本）：
```bash
# 对于完整版RKNN Toolkit 2
pip install rknn-toolkit2

# 或者轻量级版本
pip install rknn-toolkit-lite2
```

## 使用方法

### 1. 单张图片检测（FP32模型）

```bash
python detect_oneimg_fp32.py --model model/yolov8n.rknn --image path/to/your/image.jpg --output output
```

参数说明：
- `--model`: RKNN模型路径
- `--image`: 要检测的图片路径
- `--output`: 输出目录
- `--conf`: 置信度阈值（默认0.65）
- `--nms`: NMS阈值（默认0.45）
- `--size`: 输入图像大小（默认640）
- `--no-label`: 不加载和显示标签文件

### 2. 单张图片检测（INT8量化模型）

```bash
python detect_oneimg_int8.py
```

使用前请在脚本中设置以下参数：
- `RKNN_MODEL`: INT8量化模型路径
- `IMG_PATH`: 测试图片路径
- `OUTPUT_DIR`: 输出目录

### 3. 测试集评估

```bash
python detect_testset_fp32.py
```

使用前请在`Config`类中设置以下参数：
- `MODEL_PATH`: 模型路径
- `TEST_DIR`: 测试集目录
- `IMAGE_DIR`: 测试图片目录
- `LABEL_DIR`: 测试标签目录

### 4. 模型性能对比

```bash
python compare.py
```

使用前请在脚本中设置以下参数：
- `RKNN_FP32_MODEL`: FP32模型路径
- `RKNN_W8A8_MODEL`: W8A8量化模型路径
- `TEST_IMAGE_PATH`: 测试图片路径

## 项目结构

```
RK3588-YOLO-Defect-Detection/
├── bin/                      # 二进制工具
│   └── query_model_attr      # 查询模型属性工具
├── model/                    # 模型文件
│   ├── yolo11n.rknn          # YOLOv8n模型（版本11）
│   ├── yolo12n.rknn          # YOLOv8n模型（版本12）
│   └── yolov8n.rknn          # YOLOv8n基础模型
├── compare.py                # 模型性能对比脚本
├── detect_oneimg_fp32.py     # FP32模型单张图片检测
├── detect_oneimg_int8.py     # INT8模型单张图片检测
├── detect_realtime_fp32.py   # 实时检测脚本
├── detect_testset_fp32.py    # 测试集评估脚本
└── README.md                 # 项目说明文档
```

## 性能指标

在RK3588平台上，使用YOLOv8n模型的典型性能：

- FP32模型：
  - 推理时间：约20-30ms/帧
  - FPS：约33-50帧/秒

- INT8量化模型：
  - 推理时间：约10-15ms/帧
  - FPS：约66-100帧/秒

实际性能可能因设备、模型大小和输入分辨率而异。

## 注意事项

1. 确保已正确安装RKNN相关工具包
2. 根据实际设备修改`DEVICE_ID`
3. 量化模型需要设置正确的量化参数
4. 推理前确保图像预处理与模型训练时一致

## 许可证

[添加您的许可证信息]

## 贡献指南

[添加贡献指南]

## 联系方式

[yyShine760@163.com]
