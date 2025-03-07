#ifndef CUDA_NETWORK_HPP
#define CUDA_NETWORK_HPP

#include <vector>
#include "cuda_layer.hpp"
#include "cuda_node.hpp"


void network::network_forward_propagationGPU(const std::vector<double> &input);

#endif