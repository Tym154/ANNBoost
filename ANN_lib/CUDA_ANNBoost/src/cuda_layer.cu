#include "../include/cuda_layer.hpp"


void network_layer::layer_forward_propagation_GPU(const std::vector<network_node> &nodes_in_previous_layer){
    int num_nodes = nodes_in_layer.size();
    int num_inputs = nodes_in_previous_layer.size();

    std::vector<double> host_weights(num_nodes * num_inputs);
    std::vector<double> host_inputs(num_inputs);
    std::vector<double> host_biases(num_nodes);
    std::vector<double> host_outputs(num_nodes);

    for(size_t i = 0; i < num_nodes; i++){
        host_biases[i] = nodes_in_layer[i].bias;

        for(size_t j = 0; j < num_inputs; j++){
            host_weights[i * num_inputs + j] = nodes_in_layer[i].input_weights[j];
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

    int activation = nodes_in_layer[0].activation_type_chosen;

    int threads_per_block = 512;
    int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    layer_forward_propagation_kernel<<<num_blocks, threads_per_block>>>(d_weights, d_inputs, d_biases, d_outputs, num_nodes, num_inputs, activation);

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
        nodes_in_layer[i].activation = host_outputs[i];
    }

    cudaFree(d_weights);
    cudaFree(d_inputs);
    cudaFree(d_biases);
    cudaFree(d_outputs);
}

__global__ void layer_forward_propagation_kernel(double* d_weights, double* d_inputs, double* d_biases, double* d_outputs, int num_nodes, int num_inputs, int activation_type){
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

std::vector<double> network_layer::layer_backward_propagation_GPU(const std::vector<double> &losses, const std::vector<network_node> &previous_layer_nodes,const double &learning_rate){
    int num_nodes = nodes_in_layer.size();
    int prev_layer_size = previous_layer_nodes.size();

    // Allocate host memory
    std::vector<float> host_previous_layer_losses(prev_layer_size, 0.0);
    std::vector<double> host_weights(num_nodes * prev_layer_size);
    std::vector<double> host_biases(num_nodes);
    std::vector<double> host_derivatives(num_nodes);
    std::vector<double> host_activations(prev_layer_size);

    // Copy weights, biases, activations, and derivatives from the layer
    for (size_t i = 0; i < num_nodes; i++) {
        for (size_t j = 0; j < prev_layer_size; j++) {
            host_weights[i * prev_layer_size + j] = nodes_in_layer[i].input_weights[j];
        }
        host_biases[i] = nodes_in_layer[i].bias;
        host_derivatives[i] = nodes_in_layer[i].get_derivative();
    }

    for (size_t j = 0; j < prev_layer_size; j++) {
        host_activations[j] = previous_layer_nodes[j].activation;
    }

    // Allocate device memory
    double *d_losses, *d_weights, *d_biases, *d_derivatives, *d_activations;
    float *d_prev_layer_losses;

    cudaMalloc(&d_weights, num_nodes * prev_layer_size * sizeof(double));
    cudaMalloc(&d_prev_layer_losses, prev_layer_size * sizeof(float));
    cudaMalloc(&d_biases, num_nodes * sizeof(double));
    cudaMalloc(&d_derivatives, num_nodes * sizeof(double));
    cudaMalloc(&d_activations, prev_layer_size * sizeof(double));
    cudaMalloc(&d_losses, losses.size() * sizeof(double));

    cudaMemcpy(d_weights, host_weights.data(), num_nodes * prev_layer_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_layer_losses, host_previous_layer_losses.data(), prev_layer_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, host_biases.data(), num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_derivatives, host_derivatives.data(), num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_activations, host_activations.data(), prev_layer_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_losses, losses.data(), losses.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel configuration
    int threads_per_block = 512;
    int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    // Launch the kernel with error checking
    compute_previous_layer_losses<<<num_blocks, threads_per_block>>>(d_losses, d_weights, d_prev_layer_losses, num_nodes, prev_layer_size);
    update_weights_biases<<<num_blocks, threads_per_block>>>(d_losses, d_activations, d_weights, d_biases, d_derivatives, learning_rate, num_nodes, prev_layer_size);
    cudaDeviceSynchronize();

    cudaMemcpy(host_weights.data(), d_weights, num_nodes * prev_layer_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_biases.data(), d_biases, num_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_previous_layer_losses.data(), d_prev_layer_losses, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // Update layer weights and biases
    for (size_t i = 0; i < num_nodes; i++) {    
        for (size_t j = 0; j < prev_layer_size; j++) {
            nodes_in_layer[i].input_weights[j] = host_weights[i * prev_layer_size + j];
        }
        nodes_in_layer[i].bias = host_biases[i];
    }

    // Free device memory
    cudaFree(d_losses);
    cudaFree(d_weights);
    cudaFree(d_prev_layer_losses);
    cudaFree(d_biases);
    cudaFree(d_derivatives);
    cudaFree(d_activations);

    // Return previous layer losses
    return std::vector<double>(host_previous_layer_losses.begin(), host_previous_layer_losses.end());
}

__global__ void compute_previous_layer_losses(double *d_losses, double *d_weights, float *d_prev_layer_losses, int num_nodes, int prev_layer_size){
    int node_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_index < num_nodes) {
        for (int j = 0; j < prev_layer_size; j++) {
            atomicAdd(&d_prev_layer_losses[j], d_losses[node_index] * d_weights[node_index * prev_layer_size + j]);
        }
    }
}

__global__ void update_weights_biases(double *d_losses, double *d_activations, double *d_weights, double *d_biases, double *d_derivatives, double learning_rate, int num_nodes, int prev_layer_size){
    int node_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_index < num_nodes) {
        double gradient = d_losses[node_index] * d_derivatives[node_index];

        for (int j = 0; j < prev_layer_size; j++) {
            d_weights[node_index * prev_layer_size + j] += learning_rate * gradient * d_activations[j];
        }

        d_biases[node_index] += learning_rate * gradient;
    }
}
