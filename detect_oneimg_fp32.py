# 检测单张图片

import cv2
import numpy as np
from rknn.api import RKNN
import os
import time
import argparse
import sys

# 默认配置，可通过命令行参数覆盖
DEFAULT_RKNN_MODEL = r''
DEFAULT_IMG_PATH = r''
DEFAULT_OUTPUT_DIR = r'output'

# 模型参数
OBJ_THRESH = 0.65  # 置信度阈值
NMS_THRESH = 0.45  # NMS阈值
IMG_SIZE = 640     # 输入图像大小

# PCB缺陷类别
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv8 单张图片检测')
    parser.add_argument('--model', type=str, default=DEFAULT_RKNN_MODEL,
                        help='RKNN模型路径')
    parser.add_argument('--image', type=str, default=DEFAULT_IMG_PATH,
                        help='要检测的图片路径')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--conf', type=float, default=OBJ_THRESH,
                        help='置信度阈值')
    parser.add_argument('--nms', type=float, default=NMS_THRESH,
                        help='NMS阈值')
    parser.add_argument('--size', type=int, default=IMG_SIZE,
                        help='输入图像大小')
    parser.add_argument('--no-label', action='store_true',
                        help='不加载和显示标签文件')
    
    return parser.parse_args()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """调整图像大小并添加填充"""
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def process_output(output, r, dwdh, obj_thresh, img_size):
    """处理模型输出"""
    try:
        # 添加输出shape的调试信息
        print(f"Output shape: {output.shape}")
        
        # 根据PCB数据集的类别调整维度
        num_classes = len(CLASSES)
        
        # 尝试不同的输出格式处理方法
        if len(output.shape) == 3 and output.shape[0] == 1:
            # 如果输出是 (1, num_classes+4, N) 格式
            outputs = np.transpose(output[0])
        elif len(output.shape) == 2:
            # 如果输出已经是 (N, num_classes+4) 格式
            outputs = output
        else:
            # 尝试重塑为预期格式
            try:
                outputs = np.transpose(output.reshape((num_classes + 4), -1))
            except Exception as e:
                print(f"警告: 输出格式处理失败: {e}")
                print(f"尝试备用处理方法...")
                # 备用处理方法
                if output.size > 0:
                    outputs = output.reshape(-1, num_classes + 4)
                else:
                    print("错误: 输出数据为空")
                    return []
        
        # 分离边界框和分类分数
        if outputs.shape[1] >= num_classes + 4:
            scores = outputs[:, 4:(4 + num_classes)]
            boxes = outputs[:, :4]
        else:
            print(f"错误: 输出维度不匹配: {outputs.shape}")
            return []
        
        # 获取每个框的最高分数和对应的类别
        max_score_indices = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(len(scores)), max_score_indices]
        
        # 应用置信度阈值
        mask = max_scores > obj_thresh
        boxes = boxes[mask]
        max_score_indices = max_score_indices[mask]
        max_scores = max_scores[mask]
        
        if len(boxes) == 0:
            return []

        # 还原边界框坐标
        boxes = np.divide(boxes, r)
        boxes = boxes - np.array(dwdh * 2)
        
        # 转换为xyxy格式
        boxes = np.concatenate([
            boxes[:, :2] - boxes[:, 2:] / 2,
            boxes[:, :2] + boxes[:, 2:] / 2
        ], axis=1)
        
        # 确保坐标在有效范围内
        boxes = np.clip(boxes, 0, None)
        
        # 应用NMS
        try:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_scores.tolist(), obj_thresh, NMS_THRESH)
            
            results = []
            for idx in indices:
                # 处理不同版本OpenCV的索引差异
                if isinstance(idx, np.ndarray):
                    idx = idx[0]
                
                box = boxes[idx]
                score = max_scores[idx]
                cls_id = max_score_indices[idx]
                results.append({
                    'box': box.astype(int).tolist(),
                    'score': float(score),
                    'class_id': int(cls_id),
                    'class_name': CLASSES[cls_id] if cls_id < len(CLASSES) else f"unknown_{cls_id}"
                })
            
            return results
        except Exception as e:
            print(f"NMS处理错误: {e}")
            return []
            
    except Exception as e:
        print(f"输出处理错误: {e}")
        return []

def load_label(image_path, classes):
    """加载标签文件"""
    # 从图片路径获取对应的标签路径
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
    print(f"尝试读取标签文件：{label_path}")
    
    boxes = []
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"标签文件内容：\n{''.join(lines[:5])}{'...' if len(lines) > 5 else ''}")
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:  # 确保格式正确
                            cls_id, x_center, y_center, w, h = map(float, parts)
                            cls_id = int(cls_id)
                            class_name = classes[cls_id] if cls_id < len(classes) else f"unknown_{cls_id}"
                            boxes.append({
                                'class_id': cls_id,
                                'class_name': class_name,
                                'box': [x_center, y_center, w, h]  # 归一化的yolo格式
                            })
                        else:
                            print(f"警告: 跳过格式错误的标签行: {line.strip()}")
                else:
                    print("标签文件为空")
        except Exception as e:
            print(f"读取标签文件出错: {e}")
    else:
        print(f"标签文件不存在: {label_path}")
    return boxes

def draw_boxes(img, boxes, color, is_normalized=True):
    """在图像上绘制检测框"""
    if not boxes:
        return
        
    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")
    
    for box in boxes:
        try:
            if is_normalized:
                # 处理归一化坐标 (YOLO格式)
                if 'box' in box and len(box['box']) == 4:
                    x_center, y_center, width, height = box['box']
                    x1 = max(0, int((x_center - width/2) * w))
                    y1 = max(0, int((y_center - height/2) * h))
                    x2 = min(w-1, int((x_center + width/2) * w))
                    y2 = min(h-1, int((y_center + height/2) * h))
                    label = f"{box['class_name']}"
                else:
                    print(f"警告: 跳过格式错误的归一化框: {box}")
                    continue
            else:
                # 处理绝对坐标 (x1, y1, x2, y2)
                if 'box' in box and len(box['box']) == 4:
                    x1, y1, x2, y2 = box['box']
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    label = f"{box['class_name']} {box.get('score', 0):.2f}"
                else:
                    print(f"警告: 跳过格式错误的检测框: {box}")
                    continue

            # 绘制边界框和标签
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"绘制框错误: {e}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 更新配置
    rknn_model = args.model
    img_path = args.image
    output_dir = args.output
    obj_thresh = args.conf
    nms_thresh = args.nms
    img_size = args.size
    load_labels = not args.no_label
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(rknn_model):
        print(f"错误: 模型文件不存在: {rknn_model}")
        return
    
    if not os.path.exists(img_path):
        print(f"错误: 图片文件不存在: {img_path}")
        return

    # 加载RKNN模型
    try:
        rknn = RKNN(verbose=True)
        print('--> 加载RKNN模型')
        ret = rknn.load_rknn(rknn_model)
        if ret != 0:
            print(f'加载RKNN模型失败! 错误码: {ret}')
            return

        # 初始化运行时环境
        print('--> 初始化运行时环境')
        ret = rknn.init_runtime(target='rk3588')
        if ret != 0:
            print(f'初始化运行时环境失败! 错误码: {ret}')
            rknn.release()
            return
    except Exception as e:
        print(f"初始化RKNN模型错误: {e}")
        return

    # 读取图片
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"读取图片失败: {img_path}")
            rknn.release()
            return
    except Exception as e:
        print(f"读取图片错误: {e}")
        rknn.release()
        return

    # 预处理
    try:
        img_resized, r, dwdh = letterbox(img, new_shape=(img_size, img_size))
        img_input = np.expand_dims(img_resized, 0)
    except Exception as e:
        print(f"图像预处理错误: {e}")
        rknn.release()
        return

    # 推理
    try:
        print('--> 运行模型')
        start = time.time()
        outputs = rknn.inference(inputs=[img_input])
        inference_time = (time.time() - start) * 1000
        print(f"推理时间: {inference_time:.2f}ms")
        
        # 添加调试信息
        if outputs and len(outputs) > 0:
            print(f"模型输出数量: {len(outputs)}")
            print(f"第一个输出形状: {outputs[0].shape}")
            print(f"输出数据类型: {outputs[0].dtype}")
            print(f"输出值范围: [{outputs[0].min()}, {outputs[0].max()}]")
        else:
            print("警告: 模型输出为空")
            rknn.release()
            return
    except Exception as e:
        print(f"模型推理错误: {e}")
        rknn.release()
        return

    # 后处理
    try:
        results = process_output(outputs[0], r, dwdh, obj_thresh, img_size)
    except Exception as e:
        print(f"后处理错误: {e}")
        results = []
    
    # 打印检测结果统计
    print("\n检测结果统计:")
    print(f"总检测目标数量: {len(results)}")
    for det in results:
        print(f"类别: {det['class_name']}, 置信度: {det['score']:.2f}, 位置: {det['box']}")

    # 加载真实标签(如果需要)
    gt_boxes = []
    if load_labels:
        try:
            gt_boxes = load_label(img_path, CLASSES)
            print("\n真实标签统计:")
            print(f"总目标数量: {len(gt_boxes)}")
            for box in gt_boxes:
                print(f"类别: {box['class_name']}, 位置: {box['box']}")
        except Exception as e:
            print(f"加载标签错误: {e}")
    
    # 创建检测结果图
    try:
        det_img = img.copy()
        draw_boxes(det_img, results, (0, 255, 0), False)
        det_path = os.path.join(output_dir, 'detection.jpg')
        cv2.imwrite(det_path, det_img)
        print(f'检测结果图像已保存到 {det_path}')
    except Exception as e:
        print(f"保存检测结果图像错误: {e}")
    
    # 创建真实标签图(如果有标签)
    if gt_boxes and load_labels:
        try:
            gt_img = img.copy()
            draw_boxes(gt_img, gt_boxes, (0, 0, 255), True)
            gt_path = os.path.join(output_dir, 'ground_truth.jpg')
            cv2.imwrite(gt_path, gt_img)
            print(f'真实标签图像已保存到 {gt_path}')
            
            # 创建对比图
            compare_img = img.copy()
            draw_boxes(compare_img, gt_boxes, (0, 0, 255), True)  # 红色为真实标签
            draw_boxes(compare_img, results, (0, 255, 0), False)  # 绿色为检测结果
            compare_path = os.path.join(output_dir, 'comparison.jpg')
            cv2.imwrite(compare_path, compare_img)
            print(f'对比图像已保存到 {compare_path}')
            
            # 计算检测统计
            gt_classes = [box['class_name'] for box in gt_boxes]
            det_classes = [box['class_name'] for box in results]
            
            print("\n检测统计:")
            print(f"真实目标数量: {len(gt_boxes)}")
            print(f"检测到的目标数量: {len(results)}")
            print("可能的漏检:", set(gt_classes) - set(det_classes))
            print("可能的误检:", set(det_classes) - set(gt_classes))
        except Exception as e:
            print(f"处理真实标签图像错误: {e}")
    
    # 释放资源
    rknn.release()
    print("\n检测完成")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"程序执行错误: {e}")
        sys.exit(1)

