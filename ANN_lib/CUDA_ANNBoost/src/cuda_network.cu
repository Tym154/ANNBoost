#include "../include/cuda_network.hpp"

void network::network_forward_propagationGPU(const std::vector<double> &input){
    for(size_t i = 0; i < layers[0].nodes_in_layer.size(); i++){
        layers[0].nodes_in_layer[i].activation = input[i];
    }

    for(size_t i = 1; i < layers.size(); i++){
        layer_forward_propagationGPU(layers[i-1].nodes_in_layer, layers[i].nodes_in_layer);
    }
}