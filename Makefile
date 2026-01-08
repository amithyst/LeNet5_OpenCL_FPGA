# ----------------------------------------------------------------------
# LeNet-5 OpenCL Makefile
# ----------------------------------------------------------------------

# Intel FPGA SDK 路径检查
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif

# 目录定义
HOST_DIR ?= host
BIN_DIR ?= bin
TARGET = $(BIN_DIR)/host

# 源文件列表：自动查找 host/src 下所有的 .cpp
SRCS := $(wildcard $(HOST_DIR)/src/*.cpp)

# 编译和链接参数
# 重点：添加 -Ihost/inc 以便找到头文件
CXX_FLAGS := -fPIC -DARM -DLINUX -I$(HOST_DIR)/inc -std=c++11

# 获取 OpenCL 编译配置
AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
# 获取 OpenCL 链接库 (针对 ARM 架构，如果是在板子上跑)
AOCL_LINK_CONFIG := $(wildcard $(INTELFPGAOCLSDKROOT)/host/arm32/lib/*.so) $(wildcard $(AOCL_BOARD_PACKAGE_ROOT)/arm32/lib/*.so) -lrt

# 编译器 (交叉编译到 DE10-Nano ARM 处理器)
CXX := arm-linux-gnueabihf-g++

# ----------------------------------------------------------------------
# 编译规则
# ----------------------------------------------------------------------

all : $(TARGET)

$(TARGET) : $(SRCS)
	@mkdir -p $(BIN_DIR)
	@echo "Compiling Host Code..."
	$(CXX) $(CXX_FLAGS) $(SRCS) -o $(TARGET) $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG)
	@echo "Build success! Output: $(TARGET)"

clean :
	@rm -f $(TARGET)
	@echo "Cleaned."