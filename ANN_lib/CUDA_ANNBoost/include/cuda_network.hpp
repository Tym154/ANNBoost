#ifndef CUDA_NETWORK_HPP
#define CUDA_NETWORK_HPP

#include "../../CORE_ANNBoost/include/core_network.hpp"
#include "cuda_layer.hpp"
#include <iostream>

void network_forward_propagation_GPU(const std::vector<double> &input);

void network_backward_propagation_GPU(const std::vector<double> &expected_activations);

void network_calculate_output_losses_GPU(const std::vector<double> &expected_activations);

__global__ void calculate_output_losses(double* d_outputs, double* d_expected_activations, double* d_activations, int num_nodes, float* latest_net_cost);

#endif
