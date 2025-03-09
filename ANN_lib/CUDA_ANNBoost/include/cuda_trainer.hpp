#ifndef CUDA_TRAINER_HPP
#define CUDA_TRAINER_HPP

#include "cuda_network.hpp"

void GPU_online_train(const std::vector<std::vector<double>> &input_data, const std::vector<std::vector<double>> &expected_activations, network& net);

#endif