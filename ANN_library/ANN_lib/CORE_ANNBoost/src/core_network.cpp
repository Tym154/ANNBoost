#include "../include/core_network.hpp"
#include <cmath>

// Basic network constructor
network::network(const std::vector<int> &layer_sizes, const activation_type &selected_hidden_activation_type, const activation_type &selected_output_activation_type, const double &selected_learning_rate, const std::pair<float, float> &bias_start_range){
    assert(layer_sizes.size() >= 2 && "Network has to have more than one layer");

    layers.reserve(layer_sizes.size());

    auto it = layer_sizes.begin();
    layers.emplace_back(network_layer(*it, 0, selected_hidden_activation_type, bias_start_range));

    // Filling the layers vector with layers
        for (++it; it != layer_sizes.end(); ++it) {
        // Determine if this is the last layer
        bool is_last_layer = (std::next(it) == layer_sizes.end());
        activation_type layer_activation = is_last_layer ? selected_output_activation_type : selected_hidden_activation_type;

        layers.emplace_back(network_layer(*it, *(it - 1), layer_activation, bias_start_range));
    }

    output_layer_losses.resize(layers.back().nodes_in_layer.size(), 0.0);
    learning_rate = selected_learning_rate; // Setting the learning rate
    hidden_activation_type_chosen = selected_hidden_activation_type; // Setting the hidden activation type
    output_activation_type_chosen = selected_output_activation_type; // Setting the output activation type
}

// Network constructor used for loading
network::network(activation_type &selected_hidden_activation_type, activation_type &selected_output_activation_type, double &selected_learning_rate, const int output_layer_size){
    learning_rate = selected_learning_rate;

    hidden_activation_type_chosen = selected_hidden_activation_type;
    output_activation_type_chosen = selected_output_activation_type;

    output_layer_losses.resize(output_layer_size, 0.0);
}

network::network(){}

// Forward propagation
void network::network_forward_propagation(const std::vector<double> &input){
    // Setting the normalized input data as activations of the input layer nodes
    for(size_t i = 0; i < layers[0].nodes_in_layer.size(); i++){
        layers[0].nodes_in_layer[i].activation = input[i];
    }

    // Performing forward propagation on layers, from input to output
    for(size_t i = 1; i < layers.size(); i++){
        layers[i].layer_forward_propagation(layers[i-1].nodes_in_layer);
    }
}

// Backward propagation
void network::network_backward_propagation(const std::vector<double> &expected_activations){
    std::vector<double> add_losses = network_calculate_output_losses(expected_activations);

    for(size_t i = 0; i < output_layer_losses.size(); i++){
        output_layer_losses[i] += add_losses[i];
    }

    // Performing backward propagation on the output layer
    std::vector<double> losses = layers.back().layer_backward_propagation(output_layer_losses, layers[layers.size() - 2].nodes_in_layer, learning_rate);

    // Performing backward propagation on the remaining layers (not including output layer)
    for(int i = layers.size()-2; i > 0; i--){
        losses = layers[i].layer_backward_propagation(losses, layers[i-1].nodes_in_layer, learning_rate);
    }

        output_layer_losses.assign(output_layer_losses.size(), 0.0);
}

// Returns the index of the most active output node
int network::get_the_most_active_in_output(){
    int most_active_index;
    double most_active = -1;

    // Iterating through the output nodes and picking the most active one
    for(size_t i = 0; i < layers.back().nodes_in_layer.size(); i++){
        if(most_active < layers.back().nodes_in_layer[i].activation){
            most_active_index = i;
            most_active = layers.back().nodes_in_layer[i].activation;
        }
    }

    // Returning the index of most active node
    return most_active_index;
}

// Returns the output layer activations
std::vector<double> network::get_output_activations() const {
    std::vector<double> output_activations(layers.back().nodes_in_layer.size(), 0.0);
    for(size_t i = 0; i < layers.back().nodes_in_layer.size(); i++){
        output_activations[i] = layers.back().nodes_in_layer[i].activation;
    }
    return output_activations;
}

// Calculates the losses of output layer (and cost of the network)
std::vector<double> network::network_calculate_output_losses(const std::vector<double> expected_activations){
    // Checking if the passed values are correct size
    assert(expected_activations.size() == layers[layers.size() - 1].nodes_in_layer.size() && "expected_activation vector has to be the same size as the output layer");
    latest_network_loss = 0.0; // Reseting the latest_network_cost to 0.0

    std::vector<double> losses(expected_activations.size(), 0);
    
    // Calculating the output layer losses
    for(size_t i = 0; i < expected_activations.size(); i++){
        // Calculating the delta of expected activation and activation of the node (which is the loss)
        losses[i] = layers[layers.size() - 1].nodes_in_layer[i].calculate_loss(expected_activations[i]);

        // Adding the loss to the cost of the network
        latest_network_loss += losses[i];
        latest_network_cost += std::pow(losses[i], 2);
    }

    return losses;
}