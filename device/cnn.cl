// ==========================================================
// 终极优化版 cnn.cl (Phase 3: Balanced & Pipelined)
// 策略: 仅展开最内层循环 (Inner Loop Unroll)
// 目标: 修复 Logic > 100% 问题，利用 Intel 编译器的流水线优化
// ==========================================================

float activation(float x) {
    return fmax(0.0f, x);
}

// ==========================================================
// Kernel 1: Conv2d (地址优化 + 局部展开)
// ==========================================================
__attribute__((num_simd_work_items(1)))
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void conv2d(
    __global const float* restrict input,
    __global const float* restrict weight,
    __global const float* restrict bias,
    __global float* restrict output,
    const int in_ch,
    const int in_size,
    const int out_ch,
    const int out_size,
    const int k_size // 假定为 5
) {
    int m = get_global_id(0); // Output Channel
    int r = get_global_id(1); // Row
    int c = get_global_id(2); // Col

    if (m >= out_ch || r >= out_size || c >= out_size) return;

    float sum = 0.0f;
    
    // 提前计算 Input 和 Weight 的部分基地址，减少循环内的乘法逻辑
    // 编译器有时无法完美提取这些，手动提取更稳健
    int w_m_offset = m * (in_ch * 25); 

    // 循环 1: 输入通道 (保持串行)
    // 展开会导致加法树过大
    for (int k = 0; k < in_ch; ++k) {
        
        int in_k_offset = k * (in_size * in_size);
        int w_k_offset  = w_m_offset + k * 25;

        // 循环 2: 卷积核行 (保持串行)
        // 关键点！不要展开这个循环。
        // 让编译器专注于流水线化，而不是并行化。
        for (int i = 0; i < 5; ++i) { 
            
            int in_r_offset = in_k_offset + (r + i) * in_size;
            int w_i_offset  = w_k_offset + i * 5;

            // 循环 3: 卷积核列 (完全展开)
            // 这是 Sweet Spot。展开 5 次是内存控制器能轻松处理的。
            // 且这 5 个地址是连续的 (c+j)，极利于 Memory Coalescing (合并访问)。
            #pragma unroll
            for (int j = 0; j < 5; ++j) {
                // 此时只剩简单的加法运算
                int in_idx = in_r_offset + (c + j);
                int w_idx  = w_i_offset + j;
                
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias[m];
    output[m * (out_size * out_size) + r * out_size + c] = activation(sum);
}

// ==========================================================
// Kernel 2: MaxPool2d (完全展开 - 保持不变)
// ==========================================================
// 2x2 = 4 次读取，非常小，完全展开没问题
__attribute__((num_simd_work_items(1))) 
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void maxpool2d(
    __global const float* restrict input,
    __global float* restrict output,
    const int in_size,
    const int out_size
) {
    int k = get_global_id(0);
    int r = get_global_id(1);
    int c = get_global_id(2);
    
    int r_start = r * 2;
    int c_start = c * 2;
    float max_val = -1e30f;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int in_idx = k * (in_size * in_size) + (r_start + i) * in_size + (c_start + j);
            float val = input[in_idx];
            if (val > max_val) max_val = val;
        }
    }

    output[k * (out_size * out_size) + r * out_size + c] = max_val;
}

// ==========================================================
// Kernel 3: FC Layer (保守展开)
// ==========================================================
__attribute__((num_simd_work_items(1)))
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void fc_layer(
    __global const float* restrict input,
    __global const float* restrict weight,
    __global const float* restrict bias,
    __global float* restrict output,
    const int in_dim,
    const int out_dim,
    const int use_activation
) {
    int m = get_global_id(0);
    if (m >= out_dim) return;

    float sum = 0.0f;
    
    // 降级: 从 unroll 16 降为 8
    // FC 层受限于 DDR 带宽，算力再快也要等内存。
    // Unroll 8 足够填满流水线，且能节省部分逻辑资源给 Conv2d 用。
    #pragma unroll 8
    for (int i = 0; i < in_dim; ++i) {
        sum += input[i] * weight[m * in_dim + i];
    }

    sum += bias[m];
    output[m] = use_activation ? activation(sum) : sum;
}