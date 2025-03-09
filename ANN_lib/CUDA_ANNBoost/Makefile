NVCC = nvcc
NVCCFLAGS = -std=c++17 -O2 -Xcompiler -fPIC

SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

LIB = libcuda_annboost.a

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -I../CORE_ANNBoost/include -c $< -o $@

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $(LIB) $(OBJS)

clean:
	rm -rf $(BUILD_DIR)/*.o $(LIB)

.PHONY: all clean
