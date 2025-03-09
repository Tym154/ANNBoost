#ifndef CORE_TRAINER_HPP
#define CORE_TRAINER_HPP

#include <vector>
#include "../include/core_network.hpp"

void online_train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations, network& net);

#endif