# 检测单张图片 (INT8 量化模型版本)

import cv2
import numpy as np
from rknn.api import RKNN
import os
import time

# --- 配置 ---
# 指向你的量化后的 RKNN 模型文件
RKNN_MODEL = r'' # 修改为量化模型路径
# 测试图片路径 
IMG_PATH = r''
# 输出目录
OUTPUT_DIR = r'' # 为 INT8 结果创建新目录

# 模型参数 (保持不变)
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

# --- 辅助函数 (letterbox, process_output, load_label, draw_boxes 基本保持不变) ---

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
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

def process_output(output, r, dwdh):
    print(f"Processing output: shape={output.shape}, dtype={output.dtype}")
    
    # YOLOv8 输出格式: (1, 10, 8400) -> (8400, 10)
    output = np.transpose(output[0])  # 转置后变为 (8400, 10)
    
    # 应用 Sigmoid 激活函数到类别分数
    scores = 1 / (1 + np.exp(-output[:, 4:]))  # 使用 sigmoid 激活
    boxes = output[:, :4]  # 前4个是边界框坐标
    
    # 获取每个框的最高分数和对应的类别
    max_scores = np.max(scores, axis=1)
    max_score_indices = np.argmax(scores, axis=1)
    
    # 应用置信度阈值
    mask = max_scores > OBJ_THRESH
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    max_score_indices = max_score_indices[mask]
    
    if len(boxes) == 0:
        return []
    
    # 还原边界框坐标（使用 sigmoid 激活）
    boxes = 1 / (1 + np.exp(-boxes))  # 使用 sigmoid 激活
    
    # 还原到原始图像尺寸
    boxes = boxes * IMG_SIZE  # 首先缩放到模型输入尺寸
    boxes = boxes / r  # 然后根据缩放比例还原
    boxes[:, [0, 2]] -= dwdh[0]  # 减去左右填充
    boxes[:, [1, 3]] -= dwdh[1]  # 减去上下填充
    
    # 转换到 xyxy 格式
    boxes = np.concatenate([
        boxes[:, :2] - boxes[:, 2:] / 2,  # xy1
        boxes[:, :2] + boxes[:, 2:] / 2   # xy2
    ], axis=1)
    
    # 确保坐标在有效范围内
    boxes = np.clip(boxes, 0, None)  # 确保坐标非负
    
    # 应用 NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_scores.tolist(), OBJ_THRESH, NMS_THRESH)
    
    results = []
    if len(indices) > 0:
        indices = indices.flatten()
        for idx in indices:
            results.append({
                'box': boxes[idx].astype(int).tolist(),
                'score': float(max_scores[idx]),
                'class_id': int(max_score_indices[idx]),
                'class_name': CLASSES[max_score_indices[idx]]
            })
    
    return results

def draw_boxes(img, boxes, color, is_normalized=True):
    """绘制检测框
    Args:
        img: 原始图像
        boxes: 检测框列表
        color: BGR颜色元组
        is_normalized: 是否是归一化坐标
    """
    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")
    
    for box_info in boxes:
        if is_normalized:
            # 处理归一化坐标 (YOLO格式: xcenter, ycenter, width, height)
            if 'box' in box_info and len(box_info['box']) == 4:
                x_center, y_center, width, height = box_info['box']
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                label = f"{box_info['class_name']}"
            else:
                print(f"警告: 跳过格式错误的归一化框: {box_info}")
                continue
        else:
            # 处理绝对坐标 (x1, y1, x2, y2)
            if 'box' in box_info and len(box_info['box']) == 4:
                x1, y1, x2, y2 = box_info['box']
                label = f"{box_info['class_name']} {box_info['score']:.2f}"
            else:
                print(f"警告: 跳过格式错误的检测框: {box_info}")
                continue

        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))

        # 绘制边界框和标签
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 在 main 函数中修改量化参数部分：
def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load RKNN model
    rknn = RKNN(verbose=True)
    print('--> Load RKNN model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print(f'Load RKNN model failed! (Error code: {ret})')
        return

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588', perf_debug=True)
    if ret != 0:
        print(f'Init runtime environment failed! (Error code: {ret})')
        rknn.release()
        return

    # Read image first
    img_orig_bgr = cv2.imread(IMG_PATH)
    if img_orig_bgr is None:
        print(f"Failed to read image: {IMG_PATH}")
        rknn.release()
        return

    # Set quantization parameters
    INPUT_ZP = -128
    INPUT_SCALE = 0.003922
    OUTPUT_ZP = -123
    OUTPUT_SCALE = 2.753625
    
    # BGR to RGB conversion
    img_orig_rgb = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    img_resized, r, dwdh = letterbox(img_orig_rgb, new_shape=(IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized, 0)  # shape: (1, 640, 640, 3)
    
    # 量化输入
    img_input_int8 = np.clip(np.round(img_input * INPUT_SCALE) + INPUT_ZP, -128, 127).astype(np.int8)

    # Inference
    print('--> Running model')
    start = time.time()
    outputs = rknn.inference(inputs=[img_input_int8])
    print(f"Inference time: {(time.time() - start)*1000:.2f}ms")

    if not outputs:
        print("Error: Inference returned empty outputs.")
        rknn.release()
        return

    # 反量化输出
    output_int8 = outputs[0]  # shape: (1, 10, 8400)
    # 先反量化，再进行后处理
    output_fp32 = (output_int8.astype(np.float32) - OUTPUT_ZP) * OUTPUT_SCALE
    
    # Post process
    results = process_output(output_fp32, r, dwdh)

    # 打印检测结果
    print("\n检测结果统计：")
    print(f"总检测目标数量: {len(results)}")
    for det in results:
        print(f"类别: {det['class_name']}, 置信度: {det['score']:.4f}, 位置: {det['box']}") # 提高置信度显示精度

    # 加载真实标签
    def load_label(image_path):
        """加载标签文件"""
        # 从图片路径获取对应的标签路径
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        print(f"读取标签文件：{label_path}")
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                print(f"标签文件内容：\n{''.join(lines)}")
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # 确保行格式正确
                        cls_id, x_center, y_center, w, h = map(float, parts)
                        boxes.append({
                            'class_id': int(cls_id),
                            'class_name': CLASSES[int(cls_id)],
                            'box': [x_center, y_center, w, h]
                        })
                    else:
                        print(f"警告: 跳过格式错误的标签行: {line.strip()}")
        else:
            print(f"警告：标签文件不存在 - {label_path}")
        return boxes
    gt_boxes = load_label(IMG_PATH)
    print("\n真实标签统计：")
    print(f"总目标数量: {len(gt_boxes)}")
    # for box in gt_boxes: # 修改变量名
    #     print(f"类别: {box['class_name']}, 位置: {box['box']}")

    # 创建真实标签图 (使用原始 BGR 图像绘制)
    gt_img = img_orig_bgr.copy() # <--- 修改这里: 使用 img_orig_bgr
    draw_boxes(gt_img, gt_boxes, (0, 0, 255), True) # 真实框用红色
    gt_path = os.path.join(OUTPUT_DIR, 'ground_truth_int8.jpg')
    cv2.imwrite(gt_path, gt_img)
    print(f'Ground truth image saved to {gt_path}')

    # 创建检测结果图 (使用原始 BGR 图像绘制)
    det_img = img_orig_bgr.copy() # <--- 修改这里: 使用 img_orig_bgr
    draw_boxes(det_img, results, (0, 255, 0), False) # 检测框用绿色
    det_path = os.path.join(OUTPUT_DIR, 'detection_int8.jpg')
    cv2.imwrite(det_path, det_img)
    print(f'Detection result saved to {det_path}')

    # 计算检测统计
    gt_classes = [box['class_name'] for box in gt_boxes]
    det_classes = [box['class_name'] for box in results]

    print("\n检测统计：")
    print(f"真实目标数量: {len(gt_boxes)}")
    print(f"检测到的目标数量: {len(results)}")
    # 简单的集合比较可能不准确，需要基于 IoU 匹配
    # print("可能的漏检：", set(gt_classes) - set(det_classes))
    # print("可能的误检：", set(det_classes) - set(gt_classes))

    rknn.release()
    print("RKNN released.")

if __name__ == '__main__':
    main()


