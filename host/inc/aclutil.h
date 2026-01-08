#ifndef ACLUTIL_H
#define ACLUTIL_H

#include "CL/opencl.h"
#include <string>
#include <vector>

// 命名空间引用，方便使用 std::string
using namespace std;

// =========================
// 内存对齐分配 (保留你原来的)
// =========================
void *acl_aligned_malloc (size_t size);
void  acl_aligned_free (void *ptr);

// =========================
// 缺失的辅助函数声明 (本次新增)
// =========================

// 设置当前工作目录为可执行文件所在的目录
bool setCwdToExeDir();

// 查找包含指定名称的 Platform
cl_platform_id findPlatform(const char *platform_name_search);

// 获取 Platform 的名字
std::string getPlatformName(cl_platform_id pid);

// 获取 Device 的名字
std::string getDeviceName(cl_device_id did);

// 获取指定 Platform 下的所有设备
cl_device_id getDevices(cl_platform_id pid, cl_device_type device_type, cl_uint *num_devices);

// 从二进制文件 (.aocx) 创建 Program
cl_program createProgramFromBinary(cl_context context, const char *binary_file_name, const cl_device_id *device_list, cl_uint num_devices);

// 检查 OpenCL 错误码，如果有错则打印并退出
void checkError(cl_int status, const char *msg);

// 获取 .aocx 文件的完整路径
std::string getBoardBinaryFile(const char *prefix, cl_device_id device);

#endif // ACLUTIL_H