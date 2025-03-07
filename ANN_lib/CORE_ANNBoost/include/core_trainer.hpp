#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <vector>
#include "../include/core_network.hpp"

void online_train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations, network& net);

#endif