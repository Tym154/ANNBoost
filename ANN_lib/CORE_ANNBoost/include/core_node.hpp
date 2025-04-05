#ifndef CORE_NODE_HPP
#define CORE_NODE_HPP

#include <vector>
#include <unordered_map>

enum activation_type {
    Sigmoid = 1,
    Relu = 2
};

class network_node{
    public:
        double value; // value of the node before activation function
        double activation; // activation of the node
        double bias; 
        std::vector<double> input_weights; // input weight from the i-th node
        activation_type activation_type_chosen; 

        // network node constructor that initializes the weights, bias and the chosen activation type
        network_node(const int &number_of_previous_nodes, const activation_type &selected_activation_type);
        // network node constructor usee for loading
        network_node(const int &number_of_previous_nodes, const activation_type &selected_activation_type, const std::vector<double> &loaded_weights, const double &loaded_bias);

        // calculating the value of the node based on the previous layer activations, their weight and a bias
        void calculate_value(const std::vector<network_node> &input_nodes);

        // activating the node based on the value of the node and the choosed activation type
        void activate_node();

        // calculating the difference between the activation of the node and the expected activation
        double calculate_loss(const int &expected_activation);

        // backward propagation to update the weights and a bias
        void node_backward_propagation(const double &loss, const std::vector<network_node> &previous_layer_nodes, const double &learning_rate);

        // returning the derivative based on the activation/value and the activation type chosen
        double get_derivative();
};

#endif