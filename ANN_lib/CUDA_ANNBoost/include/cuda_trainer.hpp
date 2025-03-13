#ifndef CUDA_TRAINER_HPP
#define CUDA_TRAINER_HPP

#include "cuda_network.hpp"

void online_train_GPU(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations, network& net);

#endif