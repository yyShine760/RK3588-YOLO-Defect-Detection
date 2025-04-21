# 对比两个 RKNN 模型的推理速度和输出差异

import os
import time
import cv2
import numpy as np
# 注意：根据您的环境，下面这行可能需要调整
# 例如，如果安装的是 rknn_toolkit_lite2 包，导入可能是 from rknn_toolkit_lite2.lite import RKNNLite
from rknnlite.api import RKNNLite # 确认这个导入路径与您的 rknnlite 安装匹配

# --- 配置参数 ---
# RKNN 模型路径 (请根据你的实际路径修改)
RKNN_FP32_MODEL = r''
RKNN_W8A8_MODEL = r''
# 测试图片路径 (请根据你的实际路径修改)
TEST_IMAGE_PATH = r''
# 模型输入尺寸 (与你转换模型时设置的一致)
MODEL_INPUT_SIZE = (640, 640) # (宽度, 高度)
# 推理运行次数 (用于计算平均时间，可适当增加)
NUM_INFERENCE_RUNS = 20
# 核心 ID (RK3588 有多个核心, 可以指定运行在哪一个上, -1 表示自动选择)
CORE_MASK = RKNNLite.NPU_CORE_AUTO # 或者 RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1 etc.

# --- 图像预处理函数 ---
def preprocess_image(image_path, target_size):
    """
    加载图像并进行预处理，使其符合 YOLOv8 输入要求 (通常是 letterbox + HWC uint8)
    RKNNLite 的 inference 通常可以直接接收 HWC uint8 格式，
    因为它在 build 模型时已经知道了 mean/std.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Cannot load image {image_path}")
        return None

    original_h, original_w = original_image.shape[:2]
    target_w, target_h = target_size

    # Letterbox resizing
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    resized_image = cv2.resize(original_image, (new_w, new_h))

    # Create a new image with target size and pad
    pad_color = [114, 114, 114] # YOLOv8 通常用灰色 (114) 填充
    padded_image = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
    padded_image[dh:dh + new_h, dw:dw + new_w, :] = resized_image

    # 返回 HWC UINT8 格式
    return padded_image

# --- 推理和计时函数 ---
def run_inference(rknn_lite, input_data, runs=10):
    """运行推理并计算平均时间"""
    outputs = None # 初始化为 None
    total_time = 0
    print(f"Warming up...")
    try:
        # 预热一次
        _ = rknn_lite.inference(inputs=[input_data]) # 输入需要是列表形式
        warmup_success = True
    except Exception as e:
        print(f"Error during warmup inference: {e}")
        # 如果预热失败，可能后续也无法进行，可以直接返回
        return None, -1 # 返回 None 表示失败，时间为 -1

    print(f"Running inference {runs} times...")
    successful_runs = 0
    for i in range(runs):
        start_time = time.time()
        try:
            # 推理！输入需要是列表形式
            current_outputs = rknn_lite.inference(inputs=[input_data])
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            successful_runs += 1
            # print(f"  Run {i+1}/{runs}: {inference_time:.4f} seconds")
            # 只需要最后一次成功运行的输出用于比较
            outputs = current_outputs
        except Exception as e:
            print(f"Error during inference run {i+1}: {e}")
            # 如果某次运行出错，可以选择跳过计时或停止
            # 这里选择继续，但这次运行不计入时间
            # 如果错误频繁发生，可能需要停止
            if i == runs - 1 and successful_runs == 0: # 如果是最后一次且从未成功过
                 outputs = None # 确保返回 None
            continue # 继续下一次循环

    if successful_runs > 0:
        avg_time = total_time / successful_runs
        print(f"Average inference time over {successful_runs} successful runs: {avg_time:.4f} seconds ({1/avg_time:.2f} FPS)")
        return outputs, avg_time
    else:
        print("No successful inference runs.")
        return None, -1 # 没有成功运行


# --- 主程序 ---
if __name__ == '__main__':

    # 1. 加载并预处理图像
    print(f"Loading and preprocessing image: {TEST_IMAGE_PATH}")
    input_image = preprocess_image(TEST_IMAGE_PATH, MODEL_INPUT_SIZE)
    if input_image is None:
        exit(1)
    print(f"Input image shape after preprocessing: {input_image.shape}, dtype: {input_image.dtype}")

    # --- >>> 添加 Batch 维度 <<< ---
    # 使用 np.expand_dims 在最前面增加一个维度 (axis=0)
    # 模型期望输入是 [Batch, Height, Width, Channels] (NHWC)
    input_image_batched = np.expand_dims(input_image, axis=0)
    print(f"Input image shape after adding batch dim: {input_image_batched.shape}, dtype: {input_image_batched.dtype}")
    # --- >>> 修改结束 <<< ---


    fp32_outputs = None
    fp32_time = -1
    w8a8_outputs = None
    w8a8_time = -1

    # 2. 测试 FP32 模型
    print("\n--- Testing FP32 Model ---")
    if os.path.exists(RKNN_FP32_MODEL):
        rknn_fp32 = RKNNLite()
        print(f"Loading FP32 model: {RKNN_FP32_MODEL}")
        ret = rknn_fp32.load_rknn(RKNN_FP32_MODEL)
        if ret != 0:
            print("Error loading FP32 RKNN model")
        else:
            print("Initializing FP32 runtime environment...")
            # 初始化时可以增加 verbose=True 获取更详细日志
            ret = rknn_fp32.init_runtime(core_mask=CORE_MASK)
            if ret != 0:
                print("Error initializing FP32 runtime")
            else:
                # --- >>> 修改：传递增加了 batch 维度的图像 <<< ---
                fp32_outputs, fp32_time = run_inference(rknn_fp32, input_image_batched, NUM_INFERENCE_RUNS)
            # 释放资源
            rknn_fp32.release()
            print("FP32 RKNNLite released.")
    else:
        print(f"FP32 model not found at {RKNN_FP32_MODEL}, skipping.")

    # 3. 测试 W8A8 模型
    print("\n--- Testing W8A8 Model ---")
    if os.path.exists(RKNN_W8A8_MODEL):
        rknn_w8a8 = RKNNLite()
        print(f"Loading W8A8 model: {RKNN_W8A8_MODEL}")
        ret = rknn_w8a8.load_rknn(RKNN_W8A8_MODEL)
        if ret != 0:
            print("Error loading W8A8 RKNN model")
        else:
            print("Initializing W8A8 runtime environment...")
             # 初始化时可以增加 verbose=True 获取更详细日志
            ret = rknn_w8a8.init_runtime(core_mask=CORE_MASK)
            if ret != 0:
                print("Error initializing W8A8 runtime")
            else:
                 # --- >>> 修改：传递增加了 batch 维度的图像 <<< ---
                w8a8_outputs, w8a8_time = run_inference(rknn_w8a8, input_image_batched, NUM_INFERENCE_RUNS)
            # 释放资源
            rknn_w8a8.release()
            print("W8A8 RKNNLite released.")
    else:
        print(f"W8A8 model not found at {RKNN_W8A8_MODEL}, skipping.")

    # 4. 结果比较
    print("\n--- Comparison Summary ---")
    # 检查时间是否有效（大于等于0，因为0也是可能的，虽然不太现实，但-1表示失败）
    if fp32_time >= 0:
        print(f"FP32 Average Inference Time: {fp32_time:.4f} seconds")
    else:
        print("FP32 Model test failed or skipped.")

    if w8a8_time >= 0:
         print(f"W8A8 Average Inference Time: {w8a8_time:.4f} seconds")
    else:
        print("W8A8 Model test failed or skipped.")

    # 只有两个时间都有效才计算速度
    if fp32_time > 0 and w8a8_time > 0: # 避免除以零
        speedup = fp32_time / w8a8_time
        print(f"Speedup (FP32 Time / W8A8 Time): {speedup:.2f}x")
    elif fp32_time >= 0 and w8a8_time >= 0:
         print("Speedup calculation skipped (one or both times are zero or invalid).")


    # 比较输出数值差异 (只有两个模型都成功输出了才比较)
    if fp32_outputs is not None and w8a8_outputs is not None:
        if len(fp32_outputs) > 0 and len(w8a8_outputs) > 0:
            # 假设主要输出在第一个 tensor
            out_fp32 = fp32_outputs[0]
            out_w8a8 = w8a8_outputs[0]

            if out_fp32.shape == out_w8a8.shape:
                # 计算绝对差异的平均值
                # 确保比较前数据类型一致，通常转为 float32 进行比较
                abs_diff = np.abs(out_fp32.astype(np.float32) - out_w8a8.astype(np.float32))
                mean_abs_diff = np.mean(abs_diff)
                max_abs_diff = np.max(abs_diff) # 也看看最大差异

                # 计算余弦相似度 (衡量方向一致性)
                # 注意：需要先将它们展平成向量
                norm_fp32 = np.linalg.norm(out_fp32)
                norm_w8a8 = np.linalg.norm(out_w8a8)
                if norm_fp32 > 1e-6 and norm_w8a8 > 1e-6: # 避免除以零
                    cos_sim = np.dot(out_fp32.flatten(), out_w8a8.flatten()) / (norm_fp32 * norm_w8a8)
                else:
                    cos_sim = 0.0 # 如果范数接近零，相似度无意义或设为0

                print(f"\nOutput Tensor Comparison (First Tensor, Shape: {out_fp32.shape}):")
                print(f"  Mean Absolute Difference: {mean_abs_diff:.6f}")
                print(f"  Max Absolute Difference:  {max_abs_diff:.6f}")
                print(f"  Cosine Similarity:        {cos_sim:.6f}")
            else:
                print("\nOutput tensor shapes differ between FP32 and W8A8 models, cannot compare directly.")
                print(f"  FP32 Output Shape: {out_fp32.shape}")
                print(f"  W8A8 Output Shape: {out_w8a8.shape}")
        else:
            print("\nModels did not produce comparable outputs (output lists are empty).")
    else:
        print("\nCannot compare outputs as one or both models failed during inference.")

    print("\nComparison finished.")