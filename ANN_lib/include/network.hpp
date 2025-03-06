#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <cassert>
#include "node.hpp"
#include "layer.hpp"

class network{
    public:
        std::vector<network_layer> layers; // layers in the network
        std::vector<double> output_layer_losses; // losses of the output layers
        double latest_network_cost; // cost of the network
        double learning_rate; // selected learning rate
        activation_type activation_type_chosen;

        // network constructor
        network(const std::vector<int> &layer_sizes, const activation_type &selected_activation_type, const double &selected_learning_rate);
        network(activation_type &selected_activation_type, double &selected_learning_rate);

        // propagating forward
        void network_forward_propagation(const std::vector<double> &input);

        void network_forward_propagationGPU(const std::vector<double> &input);

        void network_backward_propagation(const std::vector<double> &expected_activations);

        int get_the_most_active_in_output();
    
    private:
        std::vector<double> network_calculate_output_losses(const std::vector<double> expected_activations);
};

#endif