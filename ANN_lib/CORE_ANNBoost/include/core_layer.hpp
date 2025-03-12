#ifndef CORE_LAYER_HPP
#define CORE_LAYER_HPP

#include <vector>
#include "core_node.hpp"
#include "core_layer.hpp"

class network_layer{
    public:
        std::vector<network_node> nodes_in_layer; 

        // network layer constructor to initialize the nodes in the layer
        network_layer(const int &number_of_nodes,const int &number_of_previous_nodes, const activation_type &selected_activation_type);
        // second constructor for loading
        network_layer();


        // forward propagation
        void layer_forward_propagation(const std::vector<network_node> &nodes_in_previous_layer);

        // backward propagation
        std::vector<double> layer_backward_propagation(const std::vector<double> &losses, const std::vector<network_node> &previous_layer_nodes,const double &learning_rate);
        
        
        // There are all the GPU functions
    public:
        void layer_forward_propagationGPU(const std::vector<network_node> &nodes_in_previous_layer);

        std::vector<double> layer_backward_propagationGPU(const std::vector<double> &losses, const std::vector<network_node> &previous_layer_nodes,const double &learning_rate);
};

#endif