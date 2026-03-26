#include "../include/core_node.hpp"
#include <cmath>
#include <cassert>

// Network node constructor that initializes the weights, bias and the chosen activation type
network_node::network_node(const int &number_of_previous_nodes, const activation_type &selected_activation_type, const std::pair<float, float> &bias_start_range): value(0.0), activation(0.0), bias(0.0){
    if(number_of_previous_nodes != 0){
        activation_type_chosen = selected_activation_type;
        for(int i = 0; i < number_of_previous_nodes; i++){
            input_weights.push_back(-5 + 10 * (static_cast<double>(rand()) / RAND_MAX));
        }

        float min = bias_start_range.first;
        float max = bias_start_range.second;
        float span = max - min;

        bias = min + (static_cast<double>(rand()) / RAND_MAX) * span;
    }
}

// Network node constructor usee for loading
network_node::network_node(const int &number_of_previous_nodes, const activation_type &selected_activation_type, const std::vector<double> &loaded_weights, const double &loaded_bias){
    if(number_of_previous_nodes != 0){
        activation_type_chosen = selected_activation_type;
        input_weights = loaded_weights;
        bias = loaded_bias;
    }
}

// Calculating the value of the node based on the previous layer activations, their weight and a bias
void network_node::calculate_value(const std::vector<network_node> &input_nodes){
    double sumation = 0;

    // Summing the activation of the previous nodes * weight connecting to the node
    for(size_t i = 0; i < input_nodes.size(); i++){
        sumation += input_nodes[i].activation * input_weights[i];
    }

    // Adding bias to value
    value = sumation + bias;
}

// Activating the node based on the value of the node and the choosed activation type
void network_node::activate_node(){
    if(activation_type_chosen == Sigmoid){
        activation = 1 / (1 + exp(-value));
    }
    else if(activation_type_chosen == Relu){
        activation = std::max(0.0, value);
    }
    else if(activation_type_chosen == LeakyReLU){
        activation = value > 0 ? value : 0.01 * value;
    }
    else if(activation_type_chosen == Tanh){
        activation = tanh(value);
    }
    else if(activation_type_chosen == Linear){
        activation = value;
    }
    else{
        assert(false && "Incorrectly chosen activation function");
    }
}

// Calculating the difference between the activation of the node and the expected activation
double network_node::calculate_loss(const int &expected_activation){
    return expected_activation - activation;
}

// Backward propagation to update the weights and a bias
void network_node::node_backward_propagation(const double &loss, const std::vector<network_node> &previous_layer_nodes, const double &learning_rate){
    double derivative = get_derivative(); // Derivative of the activation function
    double gradient = loss * derivative; // Calculating the gradient

    // Adjusting the weights
    for(size_t i = 0; i < input_weights.size(); i++){
        input_weights[i] += learning_rate * gradient * previous_layer_nodes[i].activation;
    }

    // Adjusting the bias
    bias += learning_rate * gradient;
}

// Returning the derivative based on the activation/value and the activation type chosen
double network_node::get_derivative(){
    if(activation_type_chosen == Sigmoid){
        return activation * (1 - activation);
    }
    else if(activation_type_chosen == Relu){
        return value > 0 ? 1.0 : 0.0;
    }
    else if(activation_type_chosen == LeakyReLU){
        return value > 0 ? 1.0 : 0.01; // alpha = 0.01
    }
    else if(activation_type_chosen == Tanh){
        double tanh_val = activation;
        return 1 - (tanh_val * tanh_val);
    }
    else if(activation_type_chosen == Linear){
        return 1.0;
    }
    else{
        return -1;
    }
}