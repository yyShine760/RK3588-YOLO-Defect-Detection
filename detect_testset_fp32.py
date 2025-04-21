# 获取运行测试集各项指标

import os
import cv2
import numpy as np
from rknn.api import RKNN
from pathlib import Path
import time
import logging
from typing import List, Tuple, Optional
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置参数
class Config:
    """项目配置类"""
    MODEL_PATH = Path(r"")
    TEST_DIR = Path(r"")
    IMAGE_DIR = TEST_DIR / "images"
    LABEL_DIR = TEST_DIR / "labels"
    TARGET_PLATFORM = "rk3588"
    DEVICE_ID = "8f611d4cc9a75d34"  # Orange Pi 5 Plus 的设备 ID
    CONF_THRESHOLD = 0.65
    IOU_THRESHOLD = 0.5
    NUM_CLASSES = 6
    IMG_SIZE = 640
    CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """使用letterbox方法调整图像大小"""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def preprocess_image(image_path: Path) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """预处理图像，使用letterbox"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_processed, r, dwdh = letterbox(img, new_shape=(Config.IMG_SIZE, Config.IMG_SIZE))
    return np.expand_dims(img_processed, 0), r, dwdh

# 模型初始化
def init_rknn(model_path: str, target: str = "rk3588", device_id: str = "8f611d4cc9a75d34") -> RKNN:
    """初始化 RKNN 模型

    Args:
        model_path (str): 模型文件路径。
        target (str): 目标平台。
        device_id (str): 设备 ID。

    Returns:
        RKNN: 初始化后的 RKNN 实例。
    """
    rknn = RKNN()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        logger.error(f"加载模型失败: {model_path}")
        raise RuntimeError("Failed to load RKNN model")

    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        logger.error("初始化运行时失败")
        raise RuntimeError("Failed to initialize runtime")
    return rknn

# 工具函数
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """非极大值抑制 (NMS)

    Args:
        boxes (np.ndarray): 边界框数组，形状 (N, 4)，[x, y, w, h]。
        scores (np.ndarray): 置信度分数数组，形状 (N,)。
        iou_threshold (float): IoU 阈值。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 过滤后的边界框和分数。
    """
    if len(boxes) == 0:
        return np.array([]), np.array([])

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep], scores[keep]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个边界框的 IoU

    Args:
        box1 (List[float]): 第一个边界框 [x, y, w, h]。
        box2 (List[float]): 第二个边界框 [x, y, w, h]。

    Returns:
        float: IoU 值。
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2
    xi1, yi1 = max(x1_min, x2_min), max(y1_min, y2_min)
    xi2, yi2 = min(x1_max, x2_max), min(y1_max, y2_max)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# 推理函数
def load_and_infer(rknn: RKNN,
                  image_dir: Path,
                  label_dir: Path,
                  conf_threshold: float = 0.3,
                  iou_threshold: float = 0.5) -> Tuple[List, List, float]:
    """加载图像并进行推理，返回预测、真实标签和平均推理时间

    Args:
        rknn (RKNN): RKNN 模型实例。
        image_dir (Path): 图像目录。
        label_dir (Path): 标签目录。
        conf_threshold (float): 置信度阈值。
        iou_threshold (float): IoU 阈值。

    Returns:
        Tuple[List, List, float]: (predictions, ground_truths, avg_infer_time)
    """
    predictions = []
    ground_truths = []
    total_infer_time = 0.0
    
    # 获取所有图片路径并创建进度条
    image_paths = list(image_dir.glob("*.jpg"))
    pbar = tqdm(image_paths, desc="处理图片")
    
    for img_path in pbar:
        try:
            # 预处理
            img_input, r, dwdh = preprocess_image(img_path)
            
            # 推理
            start_time = time.time()
            outputs = rknn.inference(inputs=[img_input])
            infer_time = time.time() - start_time
            total_infer_time += infer_time
        finally:
            if outputs is None:
                logger.warning(f"推理失败: {img_path}")
                continue

        output = outputs[0].transpose(0, 2, 1)[0]
        boxes = output[:, :4]
        class_probs = output[:, 4:]
        scores = np.max(class_probs, axis=1)
        classes = np.argmax(class_probs, axis=1)

        mask = scores > conf_threshold
        pred_boxes = boxes[mask]
        pred_scores = scores[mask]
        pred_classes = classes[mask]

        if len(pred_boxes) > 0:
            pred_boxes, pred_scores = nms(pred_boxes, pred_scores, iou_threshold)
            pred_classes = pred_classes[:len(pred_boxes)]
        logger.info(f"Predictions: {len(pred_boxes)} boxes, classes: {pred_classes}, scores: {pred_scores}")
        predictions.append((pred_boxes, pred_scores, pred_classes))

        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = []
        gt_classes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    gt_boxes.append([x * 640, y * 640, w * 640, h * 640])
                    gt_classes.append(int(class_id))
        logger.info(f"Ground truths: {len(gt_boxes)} boxes, classes: {gt_classes}")
        ground_truths.append((gt_boxes, gt_classes))

    avg_infer_time = total_infer_time / len(image_paths)
    return predictions, ground_truths, avg_infer_time

# 指标计算函数
def compute_tp_fp_fn(predictions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                     ground_truths: List[Tuple[List[List[float]], List[int]]],
                     iou_threshold: float) -> Tuple[int, int, int, List[float]]:
    """计算 TP、FP、FN 和 IoU 列表

    Args:
        predictions (List[Tuple]): 预测结果列表 [(boxes, scores, classes), ...]。
        ground_truths (List[Tuple]): 真实标签列表 [(boxes, classes), ...]。
        iou_threshold (float): IoU 阈值。

    Returns:
        Tuple[int, int, int, List[float]]: (tp, fp, fn, ious)
    """
    tp, fp, fn = 0, 0, 0
    all_ious = []

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes, pred_scores, pred_classes = pred
        gt_boxes, gt_classes = gt
        matched = set()

        if len(pred_boxes) > 0:
            order = np.argsort(pred_scores)[::-1]
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_classes = pred_classes[order]

        for i, (p_box, p_score, p_class) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
            best_iou = 0
            best_idx = -1
            for j, (g_box, g_class) in enumerate(zip(gt_boxes, gt_classes)):
                if j in matched or p_class != g_class:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold:
                tp += 1
                matched.add(best_idx)
                all_ious.append(best_iou)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    return tp, fp, fn, all_ious

def compute_precision(tp: int, fp: int) -> float:
    """计算 Precision

    Args:
        tp (int): True Positives。
        fp (int): False Positives。

    Returns:
        float: Precision 值。
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def compute_recall(tp: int, fn: int) -> float:
    """计算 Recall

    Args:
        tp (int): True Positives。
        fn (int): False Negatives。

    Returns:
        float: Recall 值。
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def compute_f1(precision: float, recall: float) -> float:
    """计算 F1 Score

    Args:
        precision (float): Precision 值。
        recall (float): Recall 值。

    Returns:
        float: F1 Score 值。
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def compute_mean_iou(ious: List[float]) -> float:
    """计算 Mean IoU

    Args:
        ious (List[float]): IoU 列表。

    Returns:
        float: Mean IoU 值。
    """
    return np.mean(ious) if ious else 0

def compute_ap(predictions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
               ground_truths: List[Tuple[List[List[float]], List[int]]],
               num_classes: int,
               iou_threshold: float) -> List[float]:
    """计算每个类别的 Average Precision (AP)

    Args:
        predictions (List[Tuple]): 预测结果列表。
        ground_truths (List[Tuple]): 真实标签列表。
        num_classes (int): 类别数。
        iou_threshold (float): IoU 阈值。

    Returns:
        List[float]: 每个类别的 AP 值。
    """
    class_detections = [[] for _ in range(num_classes)]
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes, pred_scores, pred_classes = pred
        gt_boxes, gt_classes = gt
        matched = set()

        if len(pred_boxes) > 0:
            order = np.argsort(pred_scores)[::-1]
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_classes = pred_classes[order]

        for i, (p_box, p_score, p_class) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
            best_iou = 0
            best_idx = -1
            for j, (g_box, g_class) in enumerate(zip(gt_boxes, gt_classes)):
                if j in matched or p_class != g_class:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold:
                class_detections[p_class].append((p_score, 1))
                matched.add(best_idx)
            else:
                class_detections[p_class].append((p_score, 0))

    aps = []
    for cls in range(num_classes):
        detections = sorted(class_detections[cls], reverse=True)
        if not detections:
            aps.append(0)
            continue
        scores, tps = zip(*detections)
        tps = np.array(tps)
        fps = 1 - tps
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        precisions = cum_tp / (cum_tp + cum_fp)
        total_gt = sum(1 for g in ground_truths for c in g[1] if c == cls)
        recalls = cum_tp / total_gt if total_gt > 0 else np.zeros_like(cum_tp)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        aps.append(ap)
    return aps

def compute_metrics(predictions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    ground_truths: List[Tuple[List[List[float]], List[int]]],
                    num_classes: int = 6,
                    base_iou_threshold: float = 0.5) -> Tuple[float, float, float, float, float, float]:
    """计算所有性能指标

    Args:
        predictions (List[Tuple]): 预测结果列表。
        ground_truths (List[Tuple]): 真实标签列表。
        num_classes (int): 类别数。
        base_iou_threshold (float): 基础 IoU 阈值。

    Returns:
        Tuple[float, float, float, float, float, float]: (precision, recall, f1, mean_iou, mAP_50, mAP_50_95)
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_per_iou = []

    for iou_threshold in iou_thresholds:
        tp, fp, fn, ious = compute_tp_fp_fn(predictions, ground_truths, iou_threshold)
        if iou_threshold == base_iou_threshold:
            precision = compute_precision(tp, fp)
            recall = compute_recall(tp, fn)
            f1 = compute_f1(precision, recall)
            mean_iou = compute_mean_iou(ious)
        aps = compute_ap(predictions, ground_truths, num_classes, iou_threshold)
        aps_per_iou.append(np.mean(aps))

    mAP_50 = aps_per_iou[0]
    mAP_50_95 = np.mean(aps_per_iou)

    return precision, recall, f1, mean_iou, mAP_50, mAP_50_95

# 主程序
def main():
    """主程序入口"""
    try:
        logger.info("开始模型评估...")
        rknn = init_rknn(str(Config.MODEL_PATH), Config.TARGET_PLATFORM, Config.DEVICE_ID)
        
        predictions, ground_truths, avg_infer_time = load_and_infer(
            rknn, Config.IMAGE_DIR, Config.LABEL_DIR, 
            Config.CONF_THRESHOLD, Config.IOU_THRESHOLD
        )
        
        metrics = compute_metrics(
            predictions, ground_truths, Config.NUM_CLASSES, Config.IOU_THRESHOLD
        )
        
        # 打印评估结果
        logger.info("\n=== 评估结果 ===")
        logger.info(f"Precision: {metrics[0]:.4f}")
        logger.info(f"Recall: {metrics[1]:.4f}")
        logger.info(f"F1 Score: {metrics[2]:.4f}")
        logger.info(f"Mean IoU: {metrics[3]:.4f}")
        logger.info(f"mAP@0.5: {metrics[4]:.4f}")
        logger.info(f"mAP@0.5:0.95: {metrics[5]:.4f}")
        logger.info(f"平均推理时间: {avg_infer_time:.4f} 秒/张")
        logger.info(f"FPS: {1/avg_infer_time:.2f}")
        
    except Exception as e:
        logger.error(f"评估过程出错: {str(e)}")
    finally:
        rknn.release()
        logger.info("评估完成")

if __name__ == "__main__":
    main()