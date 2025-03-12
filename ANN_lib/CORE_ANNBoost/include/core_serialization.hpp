#ifndef CORE_SERIALIZATION_HPP
#define CORE_SERIALIZATION_HPP

#include "core_network.hpp"
#include <fstream>

// Saving a neural network to a txt file
void save_current_network_to_file(const network &network_needed_to_be_saved, const std::string name_of_saved_file);

// Loads network from a file
network load_network_from_file(std::string saved_network_path);

#endif 