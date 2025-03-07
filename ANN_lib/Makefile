# Set compiler
CXX = g++
NVCC = nvcc

# Options
USE_CUDA ?= 0

# Directories
CORE_DIR = CORE_ANNBoost
CPU_DIR = CPU_ANNBoost
CUDA_DIR = CUDA_ANNBoost

all: core cpu $(if $(filter 1,$(USE_CUDA)),cuda,)

core:
	$(MAKE) -C $(CORE_DIR)

cpu: core
	$(MAKE) -C $(CPU_DIR)

cuda: core
	$(MAKE) -C $(CUDA_DIR)

clean:
	$(MAKE) -C $(CORE_DIR) clean
	$(MAKE) -C $(CPU_DIR) clean
	$(MAKE) -C $(CUDA_DIR) clean

rebuild: clean all

.PHONY: all core cpu cuda clean rebuild
