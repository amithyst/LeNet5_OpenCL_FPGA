#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring> // for strcmp

#include "CL/opencl.h"
#include "aclutil.h"
#include "timer.h"
#include "mnist_loader.h"

using namespace std;

// ========================================================
// 全局 OpenCL 对象
// ========================================================
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel k_conv2d, k_maxpool2d, k_fc;

// 内存对象 (全局复用)
cl_mem d_input;
cl_mem d_w_c1, d_b_c1, d_w_c3, d_b_c3, d_w_c5, d_b_c5, d_w_f6, d_b_f6, d_w_out, d_b_out;
cl_mem d_c1_out, d_s2_out, d_c3_out, d_s4_out, d_c5_out, d_f6_out, d_final;

// ========================================================
// 辅助工具
// ========================================================

bool init_opencl() {
    cl_int status;
    // 移除 setCwdToExeDir，依靠用户在正确目录运行
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    if (!platform) return false;
    device = getDevices(platform, CL_DEVICE_TYPE_ALL, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    
    // 加载二进制 (路径相对于项目根目录)
    string bin_file = getBoardBinaryFile("bin/cnn", device);
    program = createProgramFromBinary(context, bin_file.c_str(), &device, 1);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);

    k_conv2d    = clCreateKernel(program, "conv2d", &status);
    checkError(status, "Create conv2d");
    k_maxpool2d = clCreateKernel(program, "maxpool2d", &status);
    checkError(status, "Create maxpool2d");
    k_fc        = clCreateKernel(program, "fc_layer", &status);
    checkError(status, "Create fc_layer");

    return true;
}

vector<float> load_weights(const string& filename, int count) {
    string path = "weights/" + filename;
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        printf("ERROR: Cannot open %s. Run gen_weights.py or check path!\n", path.c_str());
        exit(-1);
    }
    vector<float> buf(count);
    file.read((char*)buf.data(), count * sizeof(float));
    return buf;
}

cl_mem create_mem_copy(const vector<float>& data) {
    cl_int status;
    cl_mem mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                data.size() * sizeof(float), (void*)data.data(), &status);
    checkError(status, "Create Mem Copy");
    return mem;
}

cl_mem create_mem_empty(int float_count) {
    cl_int status;
    cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                float_count * sizeof(float), NULL, &status);
    checkError(status, "Create Mem Empty");
    return mem;
}

void softmax(vector<float>& input) {
    float max_val = *max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (float &val : input) {
        val = exp(val - max_val);
        sum += val;
    }
    for (float &val : input) val /= sum;
}

vector<float> pad_input(const vector<float>& raw, int img_idx) {
    vector<float> padded(32 * 32, 0.0f);
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            padded[(r + 2) * 32 + (c + 2)] = raw[img_idx * 784 + r * 28 + c];
        }
    }
    return padded;
}

// ========================================================
// 核心推理函数
// ========================================================
void run_inference_pass() {
    int arg;
    size_t global_ws[3];

    // Step 1: C1
    arg=0;
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_input);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_w_c1);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_b_c1);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_c1_out);
    int c1_args[] = {1, 32, 6, 28, 5};
    for(int i=0; i<5; i++) clSetKernelArg(k_conv2d, arg++, sizeof(int), &c1_args[i]);
    global_ws[0]=6; global_ws[1]=28; global_ws[2]=28;
    clEnqueueNDRangeKernel(queue, k_conv2d, 3, NULL, global_ws, NULL, 0, NULL, NULL);

    // Step 2: S2
    arg=0;
    clSetKernelArg(k_maxpool2d, arg++, sizeof(cl_mem), &d_c1_out);
    clSetKernelArg(k_maxpool2d, arg++, sizeof(cl_mem), &d_s2_out);
    int s2_args[] = {28, 14};
    clSetKernelArg(k_maxpool2d, arg++, sizeof(int), &s2_args[0]);
    clSetKernelArg(k_maxpool2d, arg++, sizeof(int), &s2_args[1]);
    global_ws[0]=6; global_ws[1]=14; global_ws[2]=14;
    clEnqueueNDRangeKernel(queue, k_maxpool2d, 3, NULL, global_ws, NULL, 0, NULL, NULL);

    // Step 3: C3
    arg=0;
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_s2_out);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_w_c3);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_b_c3);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_c3_out);
    int c3_args[] = {6, 14, 16, 10, 5};
    for(int i=0; i<5; i++) clSetKernelArg(k_conv2d, arg++, sizeof(int), &c3_args[i]);
    global_ws[0]=16; global_ws[1]=10; global_ws[2]=10;
    clEnqueueNDRangeKernel(queue, k_conv2d, 3, NULL, global_ws, NULL, 0, NULL, NULL);

    // Step 4: S4
    arg=0;
    clSetKernelArg(k_maxpool2d, arg++, sizeof(cl_mem), &d_c3_out);
    clSetKernelArg(k_maxpool2d, arg++, sizeof(cl_mem), &d_s4_out);
    int s4_args[] = {10, 5};
    clSetKernelArg(k_maxpool2d, arg++, sizeof(int), &s4_args[0]);
    clSetKernelArg(k_maxpool2d, arg++, sizeof(int), &s4_args[1]);
    global_ws[0]=16; global_ws[1]=5; global_ws[2]=5;
    clEnqueueNDRangeKernel(queue, k_maxpool2d, 3, NULL, global_ws, NULL, 0, NULL, NULL);

    // Step 5: C5
    arg=0;
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_s4_out);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_w_c5);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_b_c5);
    clSetKernelArg(k_conv2d, arg++, sizeof(cl_mem), &d_c5_out);
    int c5_args[] = {16, 5, 120, 1, 5};
    for(int i=0; i<5; i++) clSetKernelArg(k_conv2d, arg++, sizeof(int), &c5_args[i]);
    global_ws[0]=120; global_ws[1]=1; global_ws[2]=1;
    clEnqueueNDRangeKernel(queue, k_conv2d, 3, NULL, global_ws, NULL, 0, NULL, NULL);

    // Step 6: F6
    arg=0;
    clSetKernelArg(k_fc, arg++, sizeof(cl_mem), &d_c5_out);
    clSetKernelArg(k_fc, arg++, sizeof(cl_mem), &d_w_f6);
    clSetKernelArg(k_fc, arg++, sizeof(cl_mem), &d_b_f6);
    clSetKernelArg(k_fc, arg++, sizeof(cl_mem), &d_f6_out);
    int f6_args[] = {120, 84, 1};
    for(int i=0; i<3; i++) clSetKernelArg(k_fc, arg++, sizeof(int), &f6_args[i]);
    global_ws[0]=84;
    clEnqueueNDRangeKernel(queue, k_fc, 1, NULL, global_ws, NULL, 0, NULL, NULL);

    // Step 7: Output
    arg=0;
    clSetKernelArg(k_fc, arg++, sizeof(cl_mem), &d_f6_out);
    clSetKernelArg(k_fc, arg++, sizeof(cl_mem), &d_w_out);
    clSetKernelArg(k_fc, arg++, sizeof(cl_mem), &d_b_out);
    clSetKernelArg(k_fc, arg++, sizeof(cl_mem), &d_final);
    int out_args[] = {84, 10, 0};
    for(int i=0; i<3; i++) clSetKernelArg(k_fc, arg++, sizeof(int), &out_args[i]);
    global_ws[0]=10;
    clEnqueueNDRangeKernel(queue, k_fc, 1, NULL, global_ws, NULL, 0, NULL, NULL);
}

// ========================================================
// 主函数
// ========================================================
int main(int argc, char* argv[]) {
    int num_to_run = 1; // 默认只跑 1 张s
    bool verbose = true; // 默认打印详细信息

    // --- 简单参数解析 ---
    if (argc > 1) {
        if (strcmp(argv[1], "-n") == 0 && argc > 2) {
            num_to_run = atoi(argv[2]);
            verbose = false; // 批量模式下，关闭逐张打印，只显示进度
        }
    }
    
    printf("Settings: Run %d image(s), Verbose: %s\n", num_to_run, verbose ? "ON" : "OFF");

    if (!init_opencl()) { printf("Init failed\n"); return -1; }
    printf("OpenCL Initialized.\n");

    // --- 1. 准备数据 (t10k 测试集) ---
    int num_imgs, r, c;
    vector<float> raw_imgs;
    // [修复点] 使用 uint8_t 匹配 mnist_loader.h 的返回值
    vector<uint8_t> labels_raw; 
    
    try {
        raw_imgs = read_mnist_images("data/MNIST/raw/t10k-images-idx3-ubyte", num_imgs, r, c);
        labels_raw = read_mnist_labels("data/MNIST/raw/t10k-labels-idx1-ubyte");
        // 注意：read_mnist_labels 只接受文件路径 1 个参数，不需要 num_imgs (它自己会读出来)
    } catch (...) {
        printf("ERROR: Cannot read MNIST t10k files. Check data/MNIST/raw/ folder.\n");
        return -1;
    }
    
    // 限制运行数量，防止超出文件范围
    if (num_to_run > (int)labels_raw.size()) num_to_run = labels_raw.size();

    // --- 2. 加载权重 ---
    auto w_c1 = load_weights("c1_weight.bin", 150);
    auto b_c1 = load_weights("c1_bias.bin", 6);
    auto w_c3 = load_weights("c3_weight.bin", 2400);
    auto b_c3 = load_weights("c3_bias.bin", 16);
    auto w_c5 = load_weights("c5_weight.bin", 48000);
    auto b_c5 = load_weights("c5_bias.bin", 120);
    auto w_f6 = load_weights("f6_weight.bin", 10080);
    auto b_f6 = load_weights("f6_bias.bin", 84);
    auto w_out = load_weights("out_weight.bin", 840);
    auto b_out = load_weights("out_bias.bin", 10);

    // --- 3. 创建设备内存 ---
    cl_int status;
    d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, 32*32*sizeof(float), NULL, &status);
    
    // 权重/偏置 (Copy Host Ptr)
    d_w_c1 = create_mem_copy(w_c1); d_b_c1 = create_mem_copy(b_c1);
    d_w_c3 = create_mem_copy(w_c3); d_b_c3 = create_mem_copy(b_c3);
    d_w_c5 = create_mem_copy(w_c5); d_b_c5 = create_mem_copy(b_c5);
    d_w_f6 = create_mem_copy(w_f6); d_b_f6 = create_mem_copy(b_f6);
    d_w_out = create_mem_copy(w_out); d_b_out = create_mem_copy(b_out);
    
    // 中间 Buffer
    d_c1_out = create_mem_empty(6 * 28 * 28);
    d_s2_out = create_mem_empty(6 * 14 * 14);
    d_c3_out = create_mem_empty(16 * 10 * 10);
    d_s4_out = create_mem_empty(16 * 5 * 5);
    d_c5_out = create_mem_empty(120);
    d_f6_out = create_mem_empty(84);
    d_final = create_mem_empty(10);

    printf("Memory Allocated. Starting Loop...\n");
    
    // --- 4. 循环测试 ---
    int correct_count = 0;
    double total_time_ms = 0.0;
    Timer t; 

    for (int i = 0; i < num_to_run; i++) {
        vector<float> input_host = pad_input(raw_imgs, i); // 取第 i 张
        int label = (int)labels_raw[i];

        // 写入设备
        clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, 32*32*sizeof(float), input_host.data(), 0, NULL, NULL);

        // 执行推理
        t.start();
        run_inference_pass();
        clFinish(queue); 
        t.stop();
        total_time_ms += (t.get_time_s() * 1000.0);

        // 读取结果
        vector<float> result(10);
        clEnqueueReadBuffer(queue, d_final, CL_TRUE, 0, 10 * sizeof(float), result.data(), 0, NULL, NULL);

        // 统计
        float max_val = -1e9;
        int pred = 0;
        for(int k=0; k<10; k++) {
            if(result[k] > max_val) {
                max_val = result[k];
                pred = k;
            }
        }

        if (pred == label) correct_count++;

        // 详细输出 vs 简略进度
        if (verbose) {
            softmax(result);
            printf("\n[Image %d] Label: %d, Pred: %d\n", i, label, pred);
            printf("Probabilities: ");
            for(int k=0; k<10; k++) printf("%d:%.2f  ", k, result[k]);
            printf("\n");
        } else {
            // 每跑完一张输出一个点，防止用户以为卡死了
            printf("."); 
            // 每 50 张换行，并显示当前进度
            if ((i+1) % 50 == 0) printf(" [%d/%d]\n", i+1, num_to_run);
            fflush(stdout); // 强制刷新缓冲区
        }
    }

    // --- 5. 最终报告 ---
    printf("\n\n=== Final Report ===\n");
    printf("Total Images : %d\n", num_to_run);
    printf("Accuracy     : %.2f%% (%d/%d)\n", (float)correct_count/num_to_run * 100.0f, correct_count, num_to_run);
    printf("Total Time   : %.2f ms\n", total_time_ms);
    printf("Avg Latency  : %.2f ms / image\n", total_time_ms / num_to_run);
    printf("FPS          : %.2f\n", 1000.0f / (total_time_ms / num_to_run));

    return 0;
}