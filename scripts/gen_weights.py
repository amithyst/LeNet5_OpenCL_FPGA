import numpy as np
import os
import struct

# 目标目录: 项目根目录/weights
OUTPUT_DIR = "../weights"

def save_bin(filename, data):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    file_path = os.path.join(OUTPUT_DIR, filename)
    # 确保是 float32 (C++ float)
    data.astype(np.float32).tofile(file_path)
    print(f"Saved {filename}: Shape {data.shape}")

def gen_lenet_weights():
    print(f"Generating dummy weights to {os.path.abspath(OUTPUT_DIR)}...")

    np.random.seed(42) # 固定随机种子，方便调试

    # --- Layer C1: Conv 6 filters, 1 input channel, 5x5 ---
    # Weight: [6, 1, 5, 5]
    save_bin("c1_weight.bin", np.random.randn(6, 1, 5, 5) * 0.1)
    save_bin("c1_bias.bin", np.zeros(6))

    # --- Layer S2: No weights (Max Pooling) ---

    # --- Layer C3: Conv 16 filters, 6 input channels, 5x5 ---
    # (简化版：全连接卷积，不使用稀疏连接表)
    # Weight: [16, 6, 5, 5]
    save_bin("c3_weight.bin", np.random.randn(16, 6, 5, 5) * 0.1)
    save_bin("c3_bias.bin", np.zeros(16))

    # --- Layer S4: No weights (Max Pooling) ---

    # --- Layer C5: Conv 120 filters, 16 input channels, 5x5 ---
    # Input is 5x5x16, Kernel is 5x5, so Output is 1x1x120
    # Weight: [120, 16, 5, 5]
    save_bin("c5_weight.bin", np.random.randn(120, 16, 5, 5) * 0.1)
    save_bin("c5_bias.bin", np.zeros(120))

    # --- Layer F6: Fully Connected 120 -> 84 ---
    # Weight: [84, 120]
    save_bin("f6_weight.bin", np.random.randn(84, 120) * 0.1)
    save_bin("f6_bias.bin", np.zeros(84))

    # --- Layer Output: Fully Connected 84 -> 10 ---
    # Weight: [10, 84]
    save_bin("out_weight.bin", np.random.randn(10, 84) * 0.1)
    save_bin("out_bias.bin", np.zeros(10))

if __name__ == "__main__":
    gen_lenet_weights()