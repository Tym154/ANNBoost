#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include "core_node.hpp"
#include "core_layer.hpp"

class network_layer{
    public:
        std::vector<network_node> nodes_in_layer; 

        // network layer constructor to initialize the nodes in the layer
        network_layer(const int &number_of_nodes,const int &number_of_previous_nodes,const activation_type &selected_activation_type);
        // second constructor for loading
        network_layer();


        // forward propagation
        void layer_forward_propagation(const std::vector<network_node> &nodes_in_previous_layer);

        void layer_forward_propagationGPU(const std::vector<network_node> &nodes_in_previous_layer);

        // backward propagation
        std::vector<double> layer_backward_propagation(const std::vector<double> &losses, const std::vector<network_node> &previous_layer_nodes,const double &learning_rate);

    private:
        
};

#endif