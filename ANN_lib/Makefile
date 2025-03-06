# Compiler and flags
CXX = g++
NVCC = nvcc  # NVIDIA CUDA Compiler

CXXFLAGS = -Wall -std=c++11
CUDAFLAGS = -std=c++11 -O2

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# Library and Object Files
LIBRARY_NAME = lib_ANN.a
CPU_OBJECTS = $(BUILD_DIR)/node.o $(BUILD_DIR)/layer.o $(BUILD_DIR)/network.o $(BUILD_DIR)/serialization.o $(BUILD_DIR)/trainer.o
CUDA_OBJECTS = $(BUILD_DIR)/cuda_backend.o

# Default: CPU-only build
OBJECTS = $(CPU_OBJECTS)

# If USE_CUDA is set, include CUDA objects
ifdef USE_CUDA
    OBJECTS += $(CUDA_OBJECTS)
    CXXFLAGS += -DUSE_CUDA
endif

# Build target
all: $(LIBRARY_NAME)

$(LIBRARY_NAME): $(OBJECTS)
	@echo "Creating library $@"
	ar rcs $@ $^

# Rule to compile CPU source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# Rule to compile CUDA source files if enabled
$(BUILD_DIR)/cuda_backend.o: $(SRC_DIR)/cuda_backend.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) -I$(INC_DIR) -c $< -o $@

# Clean up the build directory
clean:
	rm -rf $(BUILD_DIR) $(LIBRARY_NAME)

# Rebuild everything
rebuild: clean all

.PHONY: all clean rebuild
