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
    network test_net({784, 100, 100, 10}, Sigmoid, 0.001);

    // Load and preprocess data...

    test_net.online_train(input_data, expected_outputs);
    test_net.save_current_network_to_file("saved_network{debug}.txt");
    return 0;
}
