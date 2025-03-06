#include "../include/network.hpp"
#include <cmath>

// basic network constructor
network::network(const std::vector<int> &layer_sizes, const activation_type &selected_activation_type, const double &selected_learning_rate){
    assert(layer_sizes.size() >= 2 && "Network has to have more than one layer");

    layers.reserve(layer_sizes.size());
    auto it = layer_sizes.begin();
    layers.emplace_back(network_layer(*it, 0, selected_activation_type));

    for(it++ ; it != layer_sizes.end(); it++){
        layers.emplace_back(network_layer(*it, *(it-1), selected_activation_type));
    }

    output_layer_losses.resize(layers.back().nodes_in_layer.size(), 0.0);

    learning_rate = selected_learning_rate;

    activation_type_chosen = selected_activation_type;
}

// network constructor used for loading
network::network(activation_type &selected_activation_type, double &selected_learning_rate){
    learning_rate = selected_learning_rate;

    activation_type_chosen = selected_activation_type;
}

void network::network_forward_propagation(const std::vector<double> &input){
    for(size_t i = 0; i < layers[0].nodes_in_layer.size(); i++){
        layers[0].nodes_in_layer[i].activation = input[i];
    }

    for(size_t i = 1; i < layers.size(); i++){
        layers[i].layer_forward_propagation(layers[i-1].nodes_in_layer);
    }
}

void network::network_forward_propagationGPU(const std::vector<double> &input){
    for(size_t i = 0; i < layers[0].nodes_in_layer.size(); i++){
        layers[0].nodes_in_layer[i].activation = input[i];
    }

    for(size_t i = 1; i < layers.size(); i++){
        layers[i].layer_forward_propagationGPU(layers[i-1].nodes_in_layer);
    }
}

void network::network_backward_propagation(const std::vector<double> &expected_activations){
    std::vector<double> add_losses = network_calculate_output_losses(expected_activations);

    for(size_t i = 0; i < output_layer_losses.size(); i++){
        output_layer_losses[i] += add_losses[i];
    }

    std::vector<double> losses = layers.back().layer_backward_propagation(output_layer_losses, layers[layers.size() - 2].nodes_in_layer, learning_rate);

    for(int i = layers.size()-2; i > 0; i--){
        losses = layers[i].layer_backward_propagation(losses, layers[i-1].nodes_in_layer, learning_rate);
    }

        output_layer_losses.assign(output_layer_losses.size(), 0.0);
}

// returns the index of the most active output node
int network::get_the_most_active_in_output(){
    int most;
    double most_active = -1;
    for(size_t i = 0; i < layers.back().nodes_in_layer.size(); i++){
        if(most_active < layers.back().nodes_in_layer[i].activation){
            most = i;
            most_active = layers.back().nodes_in_layer[i].activation;
        }
    }

    return most;
}

std::vector<double> network::network_calculate_output_losses(const std::vector<double> expected_activations){
    assert(expected_activations.size() == layers[layers.size() - 1].nodes_in_layer.size() && "expected_activation vector has to be the same size as the output layer");
    latest_network_cost = 0.0;

    std::vector<double> losses(expected_activations.size(), 0);
    for(size_t i = 0; i < expected_activations.size(); i++){
        losses[i] = layers[layers.size() - 1].nodes_in_layer[i].calculate_delta(expected_activations[i]);
        latest_network_cost += losses[i];
    }

    return losses;
}