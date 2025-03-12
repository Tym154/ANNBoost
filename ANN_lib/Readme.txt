You can build this library with two parameters:
    1. USE_CUDA
    2. USE_CPU

USE_CUDA builds the GPU_ANNBoost folder too, so you can use GPU paralellization functions

USE_CPU builds the CPU_ANNBoost folder, same as the USE_CUDA but with CPU parallelization

rebuild cleans all build folders and .a files and rebuilds them from scratch

samples:

$ make USE_CUDA=1

$ make USE_CPU=1

$ make USE_CUDA=1 USE_CPU=1

$ make rebuild USE_CUDA=1

