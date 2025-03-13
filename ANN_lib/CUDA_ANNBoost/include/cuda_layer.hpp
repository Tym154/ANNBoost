#ifndef CUDA_LAYER_HPP
#define CUDA_LAYER_HPP

#include  "../../CORE_ANNBoost/include/core_layer.hpp"
#include <iostream>
#include <cuda_runtime.h>


void layer_forward_propagation_GPU(const std::vector<network_node> &nodes_in_previous_layer, std::vector<network_node> &current_layer_nodes);

__global__ void layer_forward_propagation_kernel(double* d_weights, double* d_inputs, double* d_biases, double* d_outputs, int num_nodes, int num_inputs, int activation_type);

std::vector<double> layer_backward_propagation_GPU(const std::vector<double> &losses, const std::vector<network_node> &previous_layer_nodes,const double &learning_rate);

__global__ void compute_previous_layer_losses(double *d_losses, double *d_weights, float *d_prev_layer_losses, int num_nodes, int prev_layer_size);

__global__ void update_weights_biases(double *d_losses, double *d_activations, double *d_weights, double *d_biases, double *d_derivatives, double learning_rate, int num_nodes, int prev_layer_size);


#endif