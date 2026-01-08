#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

// 字节序转换 (Big Endian -> Little Endian)
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

// 读取 MNIST 标签 (Labels)
// 返回: vector<uint8_t> (0-9 的数字)
std::vector<uint8_t> read_mnist_labels(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open label file: " << full_path << std::endl;
        exit(-1);
    }

    uint32_t magic_number = 0;
    uint32_t number_of_items = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_items, sizeof(number_of_items));

    magic_number = swap_endian(magic_number);
    number_of_items = swap_endian(number_of_items);

    if (magic_number != 2049) {
        std::cerr << "ERROR: Invalid magic number in label file!" << std::endl;
        exit(-1);
    }

    std::vector<uint8_t> labels(number_of_items);
    file.read((char*)labels.data(), number_of_items);
    return labels;
}

// 读取 MNIST 图片 (Images)
// 为了适应 OpenCL 计算，我们将像素归一化为 0.0 - 1.0 的 float
// 返回: 一维 float 数组 (total_images * 28 * 28)
std::vector<float> read_mnist_images(const std::string& full_path, int& num_images, int& rows, int& cols) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open image file: " << full_path << std::endl;
        exit(-1);
    }

    uint32_t magic_number = 0;
    uint32_t number_of_items = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_items, sizeof(number_of_items));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));

    magic_number = swap_endian(magic_number);
    num_images = swap_endian(number_of_items);
    rows = swap_endian(n_rows);
    cols = swap_endian(n_cols);

    if (magic_number != 2051) {
        std::cerr << "ERROR: Invalid magic number in image file!" << std::endl;
        exit(-1);
    }

    // 此时我们要把输入 Resize 到 LeNet-5 需要的 32x32 吗？
    // PPT 里写的是 32x32 输入，但 MNIST 原图是 28x28。
    // 简单起见，我们先读 28x28，在 Kernel 里做 Padding 或者 Host 端补 0。
    // 这里先只做纯读取 (28x28)

    int total_pixels = num_images * rows * cols;
    std::vector<unsigned char> raw_pixels(total_pixels);
    file.read((char*)raw_pixels.data(), total_pixels);

    // 转换为 float 并归一化
    std::vector<float> images_float(total_pixels);
    for(int i=0; i<total_pixels; ++i) {
        images_float[i] = raw_pixels[i] / 255.0f;
    }

    printf("[Data] Loaded %d images (%dx%d)\n", num_images, rows, cols);
    return images_float;
}

#endif