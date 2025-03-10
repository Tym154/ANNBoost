#include "../include/cuda_network.hpp"
#include "cuda_runtime.h"

void network::network_forward_propagationGPU(const std::vector<double> &input){
    for(size_t i = 0; i < layers[0].nodes_in_layer.size(); i++){
        layers[0].nodes_in_layer[i].activation = input[i];
    }

    for(size_t i = 1; i < layers.size(); i++){
        layer_forward_propagationGPU(layers[i-1].nodes_in_layer, layers[i].nodes_in_layer);
    }
}

void network::network_backward_propagationGPU(const std::vector<double> &expected_activations){
    network_calculate_output_lossesGPU(expected_activations);


}

void network::network_calculate_output_lossesGPU(const std::vector<double> &expected_activations){
    output_layer_losses.assign(output_layer_losses.size(), 0.0);

    int num_nodes = layers[layers.size() - 1].nodes_in_layer.size();

    std::vector<double> host_expected(num_nodes);
    std::vector<double> host_activation(num_nodes);
    std::vector<double> host_output(num_nodes);
    float host_latest_network_cost = 0.0;

    for(int i = 0; i < num_nodes; i++){
        host_expected[i] = expected_activations[i];
        host_activation[i] = layers[layers.size() - 1].nodes_in_layer[i].activation;
    }

    size_t expected_size = num_nodes * sizeof(double);
    size_t activation_size = num_nodes * sizeof(double);
    size_t output_size = num_nodes * sizeof(double);

    double *d_expected_activations, *d_activations, *d_outputs;
    float *d_latest_net_cost;

    cudaMalloc(&d_expected_activations, expected_size);
    cudaMalloc(&d_activations, activation_size);
    cudaMalloc(&d_outputs, output_size);
    cudaMalloc(&d_latest_net_cost, sizeof(float));

    cudaMemcpy(d_expected_activations, host_expected.data(), expected_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_activations, host_activation.data(), activation_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputs, host_output.data(), output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_latest_net_cost, &host_latest_network_cost, sizeof(float), cudaMemcpyHostToDevice);


    int threads_per_block = 512;
    int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    latest_network_cost = 0;
    calculate_output_losses<<<num_blocks, threads_per_block>>>(d_outputs, d_expected_activations, d_activations, num_nodes, d_latest_net_cost);

    cudaDeviceSynchronize();

    cudaMemcpy(&host_latest_network_cost, d_latest_net_cost, sizeof(float), cudaMemcpyDeviceToHost);
    latest_network_cost = double(host_latest_network_cost);

    cudaMemcpy(host_output.data(), d_outputs, output_size, cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < output_layer_losses.size(); i++){
        output_layer_losses[i] += host_output[i];
    }

    cudaFree(d_expected_activations);
    cudaFree(d_activations);
    cudaFree(d_outputs);
    cudaFree(d_latest_net_cost);
}

__global__ void calculate_output_lossesGPU(double* d_outputs, double* d_expected_activations, double* d_activations, int num_nodes, float* latest_net_cost){
    int node_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(node_index < num_nodes){
        d_outputs[node_index] = d_expected_activations[node_index] - d_activations[node_index];
        // Works only for GPU's with compute capability of 6 and higher
        atomicAdd(latest_net_cost, fabs(d_outputs[node_index])); 
    }
}