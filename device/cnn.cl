// ==========================================================
// 终极优化版 cnn.cl (Phase 3: Balanced & Pipelined)
// 策略: 仅展开最内层循环 (Inner Loop Unroll)
// 目标: 修复 Logic > 100% 问题，利用 Intel 编译器的流水线优化
// ==========================================================

float activation(float x) {
    return fmax(0.0f, x);
}

// ==========================================================
// Kernel 1: Conv2d v2.2 (Fixed Logic Usage)
// 策略: 保持 5 个乘法器 (v2.0水平)，但优化内存访问模式
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
    const int k_size_arg 
) {
    int m = get_global_id(0); 
    int r = get_global_id(1); 
    int c = get_global_id(2); 

    if (m >= out_ch || r >= out_size || c >= out_size) return;

    float sum = 0.0f;
    int w_m_offset = m * (in_ch * 25);

    // 循环 K: 串行
    for (int k = 0; k < in_ch; ++k) {
        int in_k_offset = k * (in_size * in_size);
        int w_k_offset = w_m_offset + k * 25;

        // 循环 I (行): 串行 <--- 关键！不要 Unroll 这里，降低 Logic
        for (int i = 0; i < 5; ++i) {
            int in_r_offset = in_k_offset + (r + i) * in_size;
            int w_i_offset = w_k_offset + i * 5;
            
            // 循环 J (列): 完全展开
            // 1. 产生 5 个并行乘法器 (Logic 可控)
            // 2. 产生连续地址访问 (input[base], input[base+1]...) -> 触发 Burst Read
            #pragma unroll
            for (int j = 0; j < 5; ++j) {
                sum += input[in_r_offset + (c + j)] * weight[w_i_offset + j];
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