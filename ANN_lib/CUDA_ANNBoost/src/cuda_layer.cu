#include "../include/cuda_layer.hpp"


void layer_forward_propagationGPU(const std::vector<network_node> &nodes_in_previous_layer, std::vector<network_node> &current_layer_nodes){
    int num_nodes = current_layer_nodes.size();
    int num_inputs = nodes_in_previous_layer.size();

    std::vector<double> host_weights(num_nodes * num_inputs);
    std::vector<double> host_inputs(num_inputs);
    std::vector<double> host_biases(num_nodes);
    std::vector<double> host_outputs(num_nodes);

    for(size_t i = 0; i < num_nodes; i++){
        host_biases[i] = current_layer_nodes[i].bias;

        for(size_t j = 0; j < num_inputs; j++){
            host_weights[i * num_inputs + j] = current_layer_nodes[i].input_weights[j];
            host_inputs[j] = nodes_in_previous_layer[j].activation;
        }
    }

    double *d_weights, *d_inputs, *d_biases, *d_outputs;
    size_t weights_size = num_nodes * num_inputs * sizeof(double);
    size_t inputs_size = num_inputs * sizeof(double);
    size_t biases_size = num_nodes * sizeof(double);
    size_t outputs_size = num_nodes * sizeof(double);

    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_inputs, inputs_size);
    cudaMalloc(&d_biases, biases_size);
    cudaMalloc(&d_outputs, outputs_size);

    cudaMemcpy(d_weights, host_weights.data(), weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, host_inputs.data(), inputs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, host_biases.data(), biases_size, cudaMemcpyHostToDevice);

    int activation = current_layer_nodes[0].activation_type_chosen;

    int threads_per_block = 512;
    int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    forward_propagation_kernel<<<num_blocks, threads_per_block>>>(d_weights, d_inputs, d_biases, d_outputs, num_nodes, num_inputs, activation);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    cudaMemcpy(host_outputs.data(), d_outputs, outputs_size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < num_nodes; i++) {
        current_layer_nodes[i].activation = host_outputs[i];
    }

    cudaFree(d_weights);
    cudaFree(d_inputs);
    cudaFree(d_biases);
    cudaFree(d_outputs);
}

__global__ void forward_propagation_kernel(double* d_weights, double* d_inputs, double* d_biases, double* d_outputs, int num_nodes, int num_inputs, int activation_type){
    int node_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_index < num_nodes) {
        double weightedSum = d_biases[node_index];

        for (int inputIndex = 0; inputIndex < num_inputs; inputIndex++) {
            weightedSum += d_weights[node_index * num_inputs + inputIndex] * d_inputs[inputIndex];
        }

        if (activation_type == 1) {  // Sigmoid
            d_outputs[node_index] = 1.0 / (1.0 + exp(-weightedSum));
        } else if (activation_type == 2) {  // ReLU
            d_outputs[node_index] = fmax(0.0, weightedSum);
        } else {
            d_outputs[node_index] = weightedSum; 
        }
    }
}