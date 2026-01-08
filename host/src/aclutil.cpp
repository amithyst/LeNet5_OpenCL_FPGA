#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "CL/opencl.h"
#include "aclutil.h"

#ifdef __linux__
#include <unistd.h>
#include <libgen.h>
#endif

using namespace std;

// ==========================================
// 1. 内存对齐 (保持不变)
// ==========================================
#define ACL_ALIGNMENT 64

#ifdef __linux__
void* acl_aligned_malloc (size_t size) {
  void *result = NULL;
  posix_memalign (&result, ACL_ALIGNMENT, size);
  return result;
}
void acl_aligned_free (void *ptr) {
  free (ptr);
}
#else // WINDOWS
#include <malloc.h>
void* acl_aligned_malloc (size_t size) {
  return _aligned_malloc (size, ACL_ALIGNMENT);
}
void acl_aligned_free (void *ptr) {
  _aligned_free (ptr);
}
#endif

// ==========================================
// 2. 错误检查 (Check Error)
// ==========================================
void checkError(cl_int status, const char *msg) {
    if(status != CL_SUCCESS) {
        printf("ERROR: %s\n", msg);
        printf("OpenCL Error Code: %d\n", status);
        exit(-1);
    }
}

// ==========================================
// 3. Platform & Device 查找
// ==========================================
std::string getPlatformName(cl_platform_id pid) {
    size_t size;
    checkError(clGetPlatformInfo(pid, CL_PLATFORM_NAME, 0, NULL, &size), "Failed to get platform name size");
    std::vector<char> name(size);
    checkError(clGetPlatformInfo(pid, CL_PLATFORM_NAME, size, name.data(), NULL), "Failed to get platform name");
    return std::string(name.data());
}

std::string getDeviceName(cl_device_id did) {
    size_t size;
    checkError(clGetDeviceInfo(did, CL_DEVICE_NAME, 0, NULL, &size), "Failed to get device name size");
    std::vector<char> name(size);
    checkError(clGetDeviceInfo(did, CL_DEVICE_NAME, size, name.data(), NULL), "Failed to get device name");
    return std::string(name.data());
}

cl_platform_id findPlatform(const char *platform_name_search) {
    cl_uint num_platforms;
    checkError(clGetPlatformIDs(0, NULL, &num_platforms), "Failed to query platforms");
    
    std::vector<cl_platform_id> platforms(num_platforms);
    checkError(clGetPlatformIDs(num_platforms, platforms.data(), NULL), "Failed to get platform IDs");

    for(const auto& pid : platforms) {
        std::string name = getPlatformName(pid);
        if(name.find(platform_name_search) != std::string::npos) {
            return pid;
        }
    }
    return NULL;
}

cl_device_id getDevices(cl_platform_id pid, cl_device_type device_type, cl_uint *num_devices) {
    cl_uint n;
    checkError(clGetDeviceIDs(pid, device_type, 0, NULL, &n), "Failed to query devices");
    
    if(num_devices) *num_devices = n;
    
    std::vector<cl_device_id> devices(n);
    checkError(clGetDeviceIDs(pid, device_type, n, devices.data(), NULL), "Failed to get device IDs");
    
    return devices[0]; // 默认返回第一个设备
}

// ==========================================
// 4. 二进制文件加载 (.aocx)
// ==========================================
// 内部辅助：读取文件二进制流
unsigned char* loadBinaryFile(const char* filename, size_t* size_ret) {
    FILE* fp = fopen(filename, "rb");
    if(!fp) return NULL;

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);

    unsigned char* result = (unsigned char*)malloc(size);
    if(result) {
        if(fread(result, 1, size, fp) != size) {
            free(result);
            fclose(fp);
            return NULL;
        }
    }
    fclose(fp);
    *size_ret = size;
    return result;
}

std::string getBoardBinaryFile(const char *prefix, cl_device_id device) {
    // 简单实现：直接返回 "bin/cnn.aocx" 这种格式
    // 实际项目中可能需要根据 device name 找 .aocx，这里简化处理
    std::string fileName = std::string(prefix) + ".aocx";
    return fileName;
}

cl_program createProgramFromBinary(cl_context context, const char *binary_file_name, const cl_device_id *device_list, cl_uint num_devices) {
    size_t binary_size;
    unsigned char* binary_data = loadBinaryFile(binary_file_name, &binary_size);
    
    if(binary_data == NULL) {
        printf("ERROR: Failed to load binary file: %s\n", binary_file_name);
        exit(-1);
    }

    cl_int status;
    cl_int kernel_status;
    cl_program program = clCreateProgramWithBinary(context, num_devices, device_list, &binary_size, (const unsigned char**)&binary_data, &kernel_status, &status);
    checkError(status, "Failed to create program with binary");
    
    free(binary_data);
    return program;
}

// ==========================================
// 5. 路径设置
// ==========================================
bool setCwdToExeDir() {
#ifdef __linux__
    // 获取当前可执行文件路径
    char path[1024];
    ssize_t count = readlink("/proc/self/exe", path, 1024);
    if(count != -1) {
        path[count] = 0; // null terminate
        char *dir = dirname(path);
        if(chdir(dir) == 0) {
            return true;
        }
    }
#endif
    // Windows 环境或失败时，默认认为已经在正确目录
    return true; 
}