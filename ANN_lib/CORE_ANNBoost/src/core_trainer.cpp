#include "../include/core_trainer.hpp"
#include <cassert>

// Trains the network one one by one without batching
void online_train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations, network& net){
    int input_data_size = input_data.size(), expected_activations_size = expected_activations.size();
    assert(input_data_size == expected_activations_size && "Input data and expected activations should have the same size");

    // Looping throught the passed data and training the network one by one without batching
    for(int i = 0; i < input_data_size && i < expected_activations_size; i++){
        net.network_forward_propagation(input_data[i]);

        net.network_backward_propagation(expected_activations[i]);
    }
}    