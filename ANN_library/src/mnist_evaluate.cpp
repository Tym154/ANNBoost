#include "core_network.hpp"
#include "core_serialization.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <cmath>

const int IMAGE_SIZE = 28 * 28;   // 784 pixels
const int NUM_CLASSES = 10;


int load_mnist_csv(const std::string& filename,
                   std::vector<std::vector<double>>& inputs,
                   std::vector<std::vector<double>>& expected,
                   std::vector<int>& labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return -1;
    }

    std::string line;
    int sample_count = 0;
    int line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() != 1 + IMAGE_SIZE) {
            if (line_number == 1 && tokens.size() > 0) {
                std::cout << "Skipping header line (non‑numeric first token)." << std::endl;
                continue;
            }
            std::cerr << "Warning: line " << line_number << " has "
                      << tokens.size() << " tokens, skipping." << std::endl;
            continue;
        }

        int label;
        try {
            label = std::stoi(tokens[0]);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing label on line " << line_number
                      << ": '" << tokens[0] << "' – skipping." << std::endl;
            continue;
        }

        if (label < 0 || label >= NUM_CLASSES) {
            std::cerr << "Warning: line " << line_number
                      << " has invalid label " << label << ", skipping." << std::endl;
            continue;
        }

        std::vector<double> input(IMAGE_SIZE);
        bool valid = true;
        for (int p = 0; p < IMAGE_SIZE; ++p) {
            try {
                int pixel = std::stoi(tokens[1 + p]);
                if (pixel < 0) pixel = 0;
                if (pixel > 255) pixel = 255;
                input[p] = pixel / 255.0;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing pixel " << p << " on line "
                          << line_number << ": '" << tokens[1 + p]
                          << "' – skipping sample." << std::endl;
                valid = false;
                break;
            }
        }
        if (!valid) continue;

        labels.push_back(label);
        std::vector<double> expected_vec(NUM_CLASSES, 0.0);
        expected_vec[label] = 1.0;
        expected.push_back(expected_vec);
        inputs.push_back(input);
        ++sample_count;
    }

    return sample_count;
}


std::pair<double, double> evaluate(network& net,
                                   const std::vector<std::vector<double>>& inputs,
                                   const std::vector<std::vector<double>>& expected,
                                   const std::vector<int>& labels) {
    if (inputs.size() != labels.size() || inputs.size() != expected.size()) {
        std::cerr << "Error: input/expected/label size mismatch." << std::endl;
        return {0.0, 0.0};
    }

    int correct = 0;
    double total_cost = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        // Forward pass
        net.network_forward_propagation(const_cast<std::vector<double>&>(inputs[i]));

        // Accuracy
        int predicted = net.get_the_most_active_in_output();
        if (predicted == labels[i]) ++correct;

        // Cost
        std::vector<double> output = net.get_output_activations();
        double sample_cost = 0.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            sample_cost -= expected[i][c] * std::log(output[c] + 1e-10); // cross-entropy
        }
        total_cost += sample_cost;
    }

    double accuracy = 100.0 * correct / inputs.size();
    double avg_cost = total_cost / inputs.size();
    return {accuracy, avg_cost};
}

// ---------------------------------------------------------------------
int main() {
    std::string base_filename = "mnist/mnist_epoch";
    std::string csv_file = "../datasets/mnist_test.csv";

    std::vector<std::vector<double>> inputs, expected;
    std::vector<int> labels;
    std::cout << "Loading MNIST data from " << csv_file << " ..." << std::endl;
    int samples = load_mnist_csv(csv_file, inputs, expected, labels);
    if (samples < 0) return -1;
    std::cout << "Loaded " << samples << " samples." << std::endl;

    if (inputs.size() > 0 && inputs[0].size() != IMAGE_SIZE) {
        std::cerr << "Warning: input size mismatch (network expects "
                  << inputs[0].size() << ", data has " << IMAGE_SIZE << ")." << std::endl;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nEpoch\tAccuracy(%)\tAvgCost\n";
    std::cout << "--------------------------------\n";

    for (int epoch = 1; epoch <= 50; ++epoch) {
        std::string net_file = base_filename + std::to_string(epoch) + ".txt";

        std::cout << "Loading " << net_file << " ..." << std::endl;
        network mnist_net;  // default constructor for loading
        try {
            mnist_net = load_network_from_file(net_file);
        } catch (const std::exception& e) {
            std::cerr << "Error loading " << net_file << ": " << e.what() << std::endl;
            continue;   // skip to next epoch
        }

        auto [accuracy, avg_cost] = evaluate(mnist_net, inputs, expected, labels);
        std::cout << epoch << "\t" << accuracy << "\t\t" << avg_cost << std::endl;
    }

    return 0;
}