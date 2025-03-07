#ifndef CUDA_LAYER_HPP
#define CUDA_LAYER_HPP

#include <iostream>
#include <cuda_runtime.h>

__global__ void forward_propagation_kernel(double* d_weights, double* d_inputs, double* d_biases, double* d_outputs, int num_nodes, int num_inputs, int activation_type);

void network_layer::layer_forward_propagationGPU(const std::vector<network_node> &nodes_in_previous_layer);

#endif