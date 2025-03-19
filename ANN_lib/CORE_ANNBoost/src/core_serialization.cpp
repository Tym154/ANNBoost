#include "../include/core_serialization.hpp"
#include <sstream>
#include <fstream>
#include <cassert>
#include <iostream>

// Saving a neural network to a txt file
void network::save_current_network_to_file(const std::string &name_of_saved_file){
    std::ofstream outputfile(name_of_saved_file);

    outputfile << activation_type_chosen << " " << learning_rate << "\n";
    for(size_t i = 0; i < layers.size() - 1; i++){
        outputfile << layers[i].nodes_in_layer.size() << " ";
    }
    outputfile << layers.back().nodes_in_layer.size();

    for(size_t i = 1; i < layers.size(); i++){
        for(size_t j = 0; j < layers[i].nodes_in_layer.size(); j++){
            outputfile << "\n";
            outputfile << layers[i].nodes_in_layer[j].bias;
            for(size_t k = 0; k < layers[i].nodes_in_layer[j].input_weights.size(); k++){
                outputfile << " " << layers[i].nodes_in_layer[j].input_weights[k];
            }
        }
    }
}

// Loads network from a file
network load_network_from_file(const std::string &saved_network_path){
    std::ifstream saved_network(saved_network_path);
    std::string parameter_line;
    getline(saved_network, parameter_line);
    std::stringstream parameters(parameter_line);

    std::string part;

    getline(parameters, part, ' ');
    activation_type loaded_activation_type = activation_type(stoi(part));

    getline(parameters, part, ' ');
    double loaded_learning_rate = stod(part);

    getline(saved_network, parameter_line);
    std::stringstream sizes_of_layers_from_file(parameter_line);
    std::string size_of_layer;
    std::vector<int> layer_sizes;
    while(getline(sizes_of_layers_from_file, size_of_layer, ' ')){
        layer_sizes.push_back(stoi(size_of_layer));
    }

    network net(loaded_activation_type, loaded_learning_rate);
    for(size_t i = 0; i < layer_sizes.size(); i++){
        net.layers.emplace_back();
        net.layers.back().nodes_in_layer.reserve(layer_sizes[i]);
    }

    std::vector<double> empty_weights;
    for (int i = 0; i < layer_sizes[0]; i++) {
        net.layers[0].nodes_in_layer.emplace_back(0, loaded_activation_type, empty_weights, 0.0);
    }

    for(size_t i = 1; i < layer_sizes.size(); i++){
        for(int j = 0; j < layer_sizes[i]; j++){
            getline(saved_network, parameter_line);
            std::stringstream parameters(parameter_line);

            std::string bias;
            getline(parameters, bias, ' ');
            double loaded_bias = stod(bias);

            std::string weight;
            std::vector<double> loaded_weights(layer_sizes[i-1], 0.0);
            for(int k = 0; k < layer_sizes[i-1]; k++){
                getline(parameters, weight, ' ');
                loaded_weights[k] = stod(weight);
            }

            net.layers[i].nodes_in_layer.emplace_back(layer_sizes[i-1], loaded_activation_type, loaded_weights, loaded_bias);
        }
    }

    net.output_layer_losses.resize(net.layers.back().nodes_in_layer.size(), 0.0);
    
    return net;
}