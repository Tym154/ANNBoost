#include "core_network.hpp"
#include "core_serialization.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iomanip>
#include <cmath>

const int IMAGE_SIZE = 32 * 32;   // 1024 (grayscale)
const int NUM_CLASSES = 10;
const int SAMPLES_PER_BATCH = 10000;


int load_cifar10_batch(const std::string& filename,
                       std::vector<std::vector<double>>& inputs,
                       std::vector<std::vector<double>>& expected,
                       std::vector<int>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return -1;
    }

    const int record_size = 1 + 32 * 32 * 3; // 3073 bytes
    std::vector<unsigned char> buffer(record_size * SAMPLES_PER_BATCH);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!file) {
        std::cerr << "Error reading " << filename << std::endl;
        return -1;
    }

    for (int i = 0; i < SAMPLES_PER_BATCH; ++i) {
        const unsigned char* record = buffer.data() + i * record_size;
        int label = static_cast<int>(record[0]);
        labels.push_back(label);

        std::vector<double> expected_vec(NUM_CLASSES, 0.0);
        expected_vec[label] = 1.0;
        expected.push_back(expected_vec);

        std::vector<double> input(IMAGE_SIZE);
        for (int p = 0; p < 1024; ++p) {
            int r = record[1 + 3 * p];
            int g = record[1 + 3 * p + 1];
            int b = record[1 + 3 * p + 2];
            input[p] = (r + g + b) / (3.0 * 255.0); // [0,1]
        }
        inputs.push_back(input);
    }
    return SAMPLES_PER_BATCH;
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

        // Cross‑entropy cost from output activations
        std::vector<double> output = net.get_output_activations();
        double sample_cost = 0.0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            sample_cost -= expected[i][c] * std::log(output[c] + 1e-10);
        }
        total_cost += sample_cost;
    }

    double accuracy = 100.0 * correct / inputs.size();
    double avg_cost = total_cost / inputs.size();
    return {accuracy, avg_cost};
}

int main() {
    std::string base_filename = "cifar_epoch";
    std::string test_file = "../datasets/cifar-10-batches-bin/test_batch.bin";

    // Load test data once
    std::vector<std::vector<double>> test_inputs, test_expected;
    std::vector<int> test_labels;
    std::cout << "Loading CIFAR‑10 test data..." << std::endl;
    int test_loaded = load_cifar10_batch(test_file, test_inputs, test_expected, test_labels);
    if (test_loaded < 0) return -1;
    std::cout << "Loaded test batch (" << test_loaded << " samples)" << std::endl;

    // Print header
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nEpoch\tAccuracy(%)\tAvgCost\n";
    std::cout << "--------------------------------\n";

    // Loop over epoch files 1 .. 100
    for (int epoch = 1; epoch <= 100; ++epoch) {
        std::string net_file = base_filename + std::to_string(epoch) + ".txt";

        std::cout << "Loading " << net_file << " ..." << std::endl;
        network cifar_net;
        try {
            cifar_net = load_network_from_file(net_file);
        } catch (const std::exception& e) {
            std::cerr << "Error loading " << net_file << ": " << e.what() << std::endl;
            continue;   // skip missing/corrupt files
        }

        auto [test_acc, test_cost] = evaluate(cifar_net, test_inputs, test_expected, test_labels);
        std::cout << epoch << "\t" << test_acc << "\t\t" << test_cost << std::endl;
    }

    return 0;
}