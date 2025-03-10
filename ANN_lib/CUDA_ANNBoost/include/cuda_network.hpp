#ifndef CUDA_NETWORK_HPP
#define CUDA_NETWORK_HPP

#include "../../CORE_ANNBoost/include/core_network.hpp"
#include "cuda_layer.hpp"
#include <iostream>

void network_forward_propagationGPU(const std::vector<double> &input);


#endif
