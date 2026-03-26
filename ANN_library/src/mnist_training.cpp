#include "core_network.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <numeric>

const int IMAGE_SIZE = 28 * 28;   // 784
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
            // Skip header line if present
            if (line_number == 1 && tokens.size() > 0) {
                std::cout << "Skipping possible header line." << std::endl;
                continue;
            }
            std::cerr << "Warning: line " << line_number << " has "
                      << tokens.size() << " tokens, skipping." << std::endl;
            continue;
        }

        // Parse label
        int label;
        try {
            label = std::stoi(tokens[0]);
        } catch (...) {
            std::cerr << "Error parsing label on line " << line_number
                      << ", skipping." << std::endl;
            continue;
        }

        if (label < 0 || label >= NUM_CLASSES) {
            std::cerr << "Warning: line " << line_number
                      << " invalid label " << label << ", skipping." << std::endl;
            continue;
        }

        // Parse pixels
        std::vector<double> input(IMAGE_SIZE);
        bool valid = true;
        for (int p = 0; p < IMAGE_SIZE; ++p) {
            try {
                int pixel = std::stoi(tokens[1 + p]);
                if (pixel < 0) pixel = 0;
                if (pixel > 255) pixel = 255;
                input[p] = pixel / 255.0;          // normalize to [0,1]
            } catch (...) {
                std::cerr << "Error parsing pixel " << p << " on line "
                          << line_number << ", skipping sample." << std::endl;
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

double evaluate(network& net,
                const std::vector<std::vector<double>>& inputs,
                const std::vector<int>& labels) {
    if (inputs.size() != labels.size()) {
        std::cerr << "Error: inputs and labels size mismatch." << std::endl;
        return 0.0;
    }
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Forward pass (may need const_cast if method lacks const correctness)
        net.network_forward_propagation(const_cast<std::vector<double>&>(inputs[i]));
        int predicted = net.get_the_most_active_in_output();
        if (predicted == labels[i]) ++correct;
    }
    return 100.0 * correct / inputs.size();
}

int main() {
    // Random number generators
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    std::random_device rd;
    std::default_random_engine rng(rd());

    std::string train_file = "../datasets/mnist_train.csv";

    std::vector<std::vector<double>> train_inputs, train_expected;
    std::vector<int> train_labels;
    std::cout << "Loading training data from " << train_file << " ..." << std::endl;
    int train_samples = load_mnist_csv(train_file, train_inputs, train_expected, train_labels);
    if (train_samples < 0) return -1;
    std::cout << "Loaded " << train_samples << " training samples." << std::endl;

    std::vector<int> layer_sizes = {IMAGE_SIZE, 256, NUM_CLASSES};
    std::pair<float, float> bias_start_range = {-0.1f, 0.1f};
    double learning_rate = 0.001;                 // initial learning rate

    network mnist_net(layer_sizes, LeakyReLU, Sigmoid, learning_rate, bias_start_range);

    // Save initial  network
    mnist_net.save_current_network_to_file("mnist_initial.txt");
    std::cout << "\nNetwork created with layers: [";
    for (size_t i = 0; i < mnist_net.layers.size(); ++i) {
        std::cout << mnist_net.layers[i].nodes_in_layer.size();
        if (i < mnist_net.layers.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Initial learning rate: " << mnist_net.learning_rate << std::endl;
    std::cout << "Hidden activation: LeakyReLU, Output activation: Sigmoid\n\n";

    int epochs = 50;
    std::cout << "Starting training for " << epochs << " epochs...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle the training data
        std::vector<int> indices(train_inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<std::vector<double>> shuffled_inputs(train_inputs.size());
        std::vector<std::vector<double>> shuffled_expected(train_expected.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            shuffled_inputs[i] = train_inputs[indices[i]];
            shuffled_expected[i] = train_expected[indices[i]];
        }

        // Train one epoch (online training over the whole shuffled set)
        std::cout << "Epoch " << epoch + 1 << " training... ";
        mnist_net.online_train(shuffled_inputs, shuffled_expected);
        std::cout << "done." << std::endl;

        // Learning rate decay (start after a few epochs)
        if (epoch >= 5) {
            mnist_net.learning_rate *= 0.95;
        }

        // Evaluate
        double test_acc = evaluate(mnist_net, train_inputs, train_labels);
        std::cout << "Epoch " << epoch + 1 << " test accuracy: " << test_acc << "%\n";
        
        std::string fname = "mnist_epoch" + std::to_string(epoch + 1) + ".txt";
        mnist_net.save_current_network_to_file(fname);
        std::cout << "Checkpoint saved to " << fname << std::endl;
        
        std::cout << std::endl;
    }

    // Save final network
    mnist_net.save_current_network_to_file("mnist_final.txt");
    std::cout << "Training finished. Final network saved to mnist_final.txt\n";

    return 0;test
}