# Compiler
CXX = g++

# Default source file (placed in SRC_DIR)
SRC_FILE ?= main.cpp

# Directories
CORE_DIR = CORE_ANNBoost
SRC_DIR  = ../src
BIN_DIR  = ../bin

# Target executable (strip extension, put in BIN_DIR)
TARGET = $(BIN_DIR)/$(basename $(notdir $(SRC_FILE)))

# Include paths
INCLUDES = -I$(CORE_DIR)/include

# Linker flags (core library)
LDFLAGS = -L$(CORE_DIR) -lcore_annboost

# Default target
all: $(TARGET)

# Build the core library and then the user program
$(TARGET): core
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/$(SRC_FILE) $(INCLUDES) $(LDFLAGS) -o $(TARGET)

# Build core library
core:
	$(MAKE) -C $(CORE_DIR)

# Clean targets
clean:
	$(MAKE) -C $(CORE_DIR) clean
	rm -rf $(BIN_DIR)

rebuild: clean all

.PHONY: all core clean rebuild