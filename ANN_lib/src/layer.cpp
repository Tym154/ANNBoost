#include "../include/layer.hpp"
#include <cmath>

network_layer::network_layer(const int &number_of_nodes,const int &number_of_previous_nodes,const activation_type &selected_activation_type){
    nodes_in_layer.reserve(number_of_nodes);
    for(int i = 0; i < number_of_nodes; i++){
        nodes_in_layer.emplace_back(network_node(number_of_previous_nodes, selected_activation_type));
    }
}

network_layer::network_layer(){}

void network_layer::layer_forward_propagation(const std::vector<network_node> &nodes_in_previous_layer){
    for(network_node &node : nodes_in_layer){
        node.calculate_value(nodes_in_previous_layer);
        node.activate_node();
    }
}

std::vector<double> network_layer::layer_backward_propagation(const std::vector<double> &losses, const std::vector<network_node> &previous_layer_nodes,const double &learning_rate){
    std::vector<double> previous_layer_losses(previous_layer_nodes.size(), 0.0);

    for(size_t i = 0; i < nodes_in_layer.size(); i++){
        for(size_t j = 0; j < previous_layer_losses.size(); j++){
            previous_layer_losses[j] += losses[i] * nodes_in_layer[i].input_weights[j];
        }

        nodes_in_layer[i].node_backward_propagation(losses[i], previous_layer_nodes, learning_rate);
    }

    return previous_layer_losses;
}


void network_layer::layer_forward_propagationGPU(const std::vector<network_node> &nodes_in_previous_layer){
    int num_nodes = nodes_in_layer.size();
    int num_inputs = nodes_in_previous_layer.size();

    std::vector<double> host_weights(num_nodes * num_inputs);
    std::vector<double> host_inputs(num_inputs);
    std::vector<double> host_biases(num_nodes);
    std::vector<double> host_outputs(num_nodes);

    for(size_t i = 0; i < num_nodes; i++){
        host_biases[i] = nodes_in_layer[i].bias;

        for(size_t j = 0; j < num_inputs; j++){
            host_weights[i * num_inputs + j] = nodes_in_layer[i].input_weights[j];
            host_inputs[j] = nodes_in_previous_layer[j].activation;
        }
    }

    double *d_weights, *d_inputs, *d_biases, *d_outputs;
    size_t weightsSize = num_nodes * num_inputs * sizeof(double);
    size_t inputsSize = num_inputs * sizeof(double);
    size_t biasesSize = num_nodes * sizeof(double);
    size_t outputsSize = num_nodes * sizeof(double);

    
}
