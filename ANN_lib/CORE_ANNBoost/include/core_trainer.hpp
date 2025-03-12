#ifndef CORE_TRAINER_HPP
#define CORE_TRAINER_HPP

#include <vector>
#include "../include/core_network.hpp"

// Trains the network one one by one without batching
void online_train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations, network& net);

#endif