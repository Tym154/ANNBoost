CXX = g++
CXXFLAGS = -std=c++17 -O2 -fPIC

SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

LIB = libcore_ann.a

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $(LIB) $(OBJS)

clean:
	rm -rf $(BUILD_DIR)/*.o $(LIB)

.PHONY: all clean
