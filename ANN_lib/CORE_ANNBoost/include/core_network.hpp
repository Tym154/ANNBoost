#ifndef CORE_NETWORK_HPP
#define CORE_NETWORK_HPP

#include <vector>
#include <cassert>
#include <iostream>
#include "core_node.hpp"
#include "core_layer.hpp"

class network{
    // There are all the CORE functions
    public:
        std::vector<network_layer> layers; // layers in the network
        std::vector<double> output_layer_losses; // losses of the output layers
        double latest_network_cost; // cost of the network
        double learning_rate; // selected learning rate
        activation_type activation_type_chosen;

        // Basic network constructors
        network(const std::vector<int> &layer_sizes, const activation_type &selected_activation_type, const double &selected_learning_rate);
        // Constructor used in loading the network
        network(activation_type &selected_activation_type, double &selected_learning_rate, const int output_layer_size);

        // Forward propagation
        void network_forward_propagation(const std::vector<double> &input);

        // Backward propagation
        void network_backward_propagation(const std::vector<double> &expected_activations);

        // Returns the index of most active node in the output layer
        int get_the_most_active_in_output();


        void online_train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations);


        void save_current_network_to_file(const std::string &name_of_saved_file);


        network load_network_from_file(const std::string &saved_network_path);
    
    private:
        // Calculates output layer losses (and cost of the network)
        std::vector<double> network_calculate_output_losses(const std::vector<double> expected_activations);


    // There are all the GPU functions
    public:
        // Forward propagation parallelized on GPU using cuda
        void network_forward_propagation_GPU(const std::vector<double> &input);

        // Backward propagation parallelized on GPU using cuda
        void network_backward_propagation_GPU(const std::vector<double> &expected_activations);


        void online_train_GPU(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations);

    private:
        // Calculates output layer losses on GPU using cuda (and cost of the network)
        void network_calculate_output_losses_GPU(const std::vector<double> &expected_activations);
};

#endif