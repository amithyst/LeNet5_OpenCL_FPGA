// 终极优化版 cnn.cl (ReLU Edition)
// 策略: 使用 ReLU 替代 Tanh 以大幅降低 Logic 占用
// 并且保持串行执行以节省硬件面积

// 移除 printf 扩展以消除警告，节省调试逻辑资源
// #pragma OPENCL EXTENSION cl_intel_printf : enable

// 核心修改: 激活函数改为 ReLU (x > 0 ? x : 0)
// 这比 tanh 省资源得多
float activation(float x) {
    return fmax(0.0f, x);
}

// ==========================================================
// Kernel 1: Conv2d (串行 + ReLU)
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
    const int k_size
) {
    int m = get_global_id(0); 
    int r = get_global_id(1); 
    int c = get_global_id(2); 

    if (m >= out_ch || r >= out_size || c >= out_size) return;

    float sum = 0.0f;

    // 保持 #pragma unroll 1 以禁止编译器自作聪明地展开循环
    #pragma unroll 1
    for (int k = 0; k < in_ch; ++k) {
        #pragma unroll 1
        for (int i = 0; i < 5; ++i) { 
            #pragma unroll 1
            for (int j = 0; j < 5; ++j) {
                int in_row = r + i;
                int in_col = c + j;
                int in_idx = k * (in_size * in_size) + in_row * in_size + in_col;
                // 权重布局: [out_ch, in_ch, k, k]
                int w_idx = m * (in_ch * 25) + k * 25 + i * 5 + j;

                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias[m];
    // 使用新的 activation (ReLU)
    output[m * (out_size * out_size) + r * out_size + c] = activation(sum);
}

// ==========================================================
// Kernel 2: MaxPool2d (保持不变)
// ==========================================================
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

    #pragma unroll 1
    for (int i = 0; i < 2; ++i) {
        #pragma unroll 1
        for (int j = 0; j < 2; ++j) {
            int in_idx = k * (in_size * in_size) + (r_start + i) * in_size + (c_start + j);
            float val = input[in_idx];
            if (val > max_val) max_val = val;
        }
    }

    output[k * (out_size * out_size) + r * out_size + c] = max_val;
}

// ==========================================================
// Kernel 3: FC Layer (串行 + ReLU)
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
    #pragma unroll 1
    for (int i = 0; i < in_dim; ++i) {
        sum += input[i] * weight[m * in_dim + i];
    }

    sum += bias[m];
    
    // 如果 use_activation 为真，应用 ReLU；否则直接输出
    output[m] = use_activation ? activation(sum) : sum;
}