#include "core_network.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <numeric>

const int IMAGE_SIZE = 32 * 32;   // 1024 
const int NUM_CLASSES = 10;
const int SAMPLES_PER_BATCH = 10000;
const int NUM_TRAIN_BATCHES = 5;
const int NUM_TEST_BATCHES = 1;

int load_cifar10_batch(const std::string& filename,
                       std::vector<std::vector<double>>& inputs,
                       std::vector<std::vector<double>>& expected,
                       std::vector<int>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return -1;
    }

    const int record_size_original = 1 + 32 * 32 * 3; // 3073 bytes
    std::vector<unsigned char> buffer(record_size_original * SAMPLES_PER_BATCH);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!file) {
        std::cerr << "Error reading " << filename << std::endl;
        return -1;
    }

    for (int i = 0; i < SAMPLES_PER_BATCH; ++i) {
        const unsigned char* record = buffer.data() + i * record_size_original;
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

double evaluate(network& net,
                const std::vector<std::vector<double>>& inputs,
                const std::vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        net.network_forward_propagation(const_cast<std::vector<double>&>(inputs[i]));
        int predicted = net.get_the_most_active_in_output();
        if (predicted == labels[i]) correct++;
    }
    return 100.0 * correct / inputs.size();
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::pair<float, float> bias_start_range = {-0.1f, 0.1f};

    network cifar_net({IMAGE_SIZE, 512, 256, NUM_CLASSES}, LeakyReLU, Sigmoid, 0.000002, bias_start_range);
    cifar_net.save_current_network_to_file("cifar_initial.txt");

    std::cout << "Layer sizes: [";
    for (size_t i = 0; i < cifar_net.layers.size(); ++i) {
        std::cout << cifar_net.layers[i].nodes_in_layer.size();
        if (i < cifar_net.layers.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Initial learning rate: " << cifar_net.learning_rate << std::endl;
    std::cout << "Hidden activation: LeakyReLU, Output activation: Sigmoid\n\n";

    std::vector<std::vector<double>> train_inputs, train_expected;
    std::vector<int> train_labels;
    std::cout << "Loading training data..." << std::endl;
    for (int i = 1; i <= NUM_TRAIN_BATCHES; ++i) {
        std::string filename = "../datasets/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin";
        int loaded = load_cifar10_batch(filename, train_inputs, train_expected, train_labels);
        if (loaded < 0) return -1;
        std::cout << "Loaded batch " << i << " (" << loaded << " samples)" << std::endl;
    }
    std::cout << "Total training samples: " << train_inputs.size() << std::endl;

    int epochs = 100;
    std::cout << "\nStarting training for " << epochs << " epochs...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<int> indices(train_inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<std::vector<double>> shuffled_inputs(train_inputs.size());
        std::vector<std::vector<double>> shuffled_expected(train_expected.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            shuffled_inputs[i] = train_inputs[indices[i]];
            shuffled_expected[i] = train_expected[indices[i]];
        }

        std::cout << "Epoch " << epoch + 1 << " training... ";
        cifar_net.online_train(shuffled_inputs, shuffled_expected);
        std::cout << "done." << std::endl;

        // Decay learning rate, starts after epoch 10
        if (epoch >= 10) cifar_net.learning_rate *= 0.94;

        // Evaluate
        double test_acc = evaluate(cifar_net, train_inputs, train_labels);
        std::cout << "Epoch " << epoch + 1 << " test accuracy: " << test_acc << "%\n";

        std::string fname = "cifar_epoch" + std::to_string(epoch + 1) + ".txt";
        cifar_net.save_current_network_to_file(fname);
    }

    cifar_net.save_current_network_to_file("cifar_final.txt");
    std::cout << "\nTraining finished. Final network saved." << std::endl;
    return 0;
    
}
