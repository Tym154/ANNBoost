# ANNBoost

**ANNBoost** is a fully custom-built, object-oriented neural network library written from scratch; with no dependencies on existing ML.

---

## Features
Implemented:
Model creation
Training loop
Model saving and loading  
Object-Oriented Design: `network`, `layer`, and `node` classes 

Not Implemented:
Logging and visualization
optimizers yet  
dataset loading utilities
---

## Project Structure

- `core_network.hpp` – core logic for model structure and forward/backward passes  
- `core_trainer.hpp` – training functions (online)  
- `core_serialization.hpp` – saving/loading (of a network)

---

## Sample Usage

Here’s an example of how to use:

```cpp
#include "core_network.hpp"
#include "core_trainer.hpp"
#include "core_serialization.hpp"

int main() {
    \\ ({input_size, {hidden_layers}, output_size}, hidden_Activation, output_activation, learning_rate, bias_init_range)
    network test_network({784, 100, 100, 10}, LeakyReLU, Sigmoid, 0.001, {-0.1f, 0.1f});

    // Load and preprocess data...

    test_network.online_train(input_data, expected_outputs);
    \\ saves to /bin
    test_network.save_current_network_to_file("saved_network.txt");
    return 0;
}
