#include "../include/cuda_trainer.hpp"

void network::online_train_GPU(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations){
    int input_data_size = input_data.size(), expected_activations_size = expected_activations.size();
    assert(input_data_size == expected_activations_size && "Input data and expected activations should have the same size");

    for(int i = 0; i < input_data_size && i < expected_activations_size; i++){
        network_forward_propagation_GPU(input_data[i]);

        network_backward_propagation(expected_activations[i]);
    }
}