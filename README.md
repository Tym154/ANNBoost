# 🧠 ANNBoost

**ANNBoost** is a fully custom-built, object-oriented neural network library written from scratch — with no dependencies on existing ML frameworks — built to support **learning and exploration** of artificial neural networks.

It’s designed to be educational, hackable, and transparent, giving you full visibility into how feedforward networks function under the hood. No black boxes. Just pure, clean neural net fundamentals.

---

## 🎯 Features

✅ From-scratch implementation (no deep learning libraries used)  
✅ Educational focus — perfect for learning and experimenting  
✅ Training loop  
✅ Optional GPU parallelization  
✅ Model saving and loading  
✅ Easy to extend with custom activation functions  
✅ Object-Oriented Design: `network`, `layer`, and `node` classes  
🚧 Logging and visualization in development  
🚫 No optimizers yet  
🚫 No dataset loading utilities (DIY-friendly)  
🚫 No regularization (yet!)

---

## 📦 Project Structure

- `core_network.hpp` – core logic for model structure and forward/backward passes  
- `core_trainer.hpp` – training functions (online/batch training)  
- `core_serialization.hpp` – tools for saving/loading network weights  
- No third-party dependencies for neural networks  
- Built-in support for MNIST CSV inputs (as shown in example)

---

## 🧪 Sample Usage

Here’s an example of how ANNBoost is used to train on a local MNIST CSV file (with added noise and normalization):

```cpp
#include "core_network.hpp"
#include "core_trainer.hpp"
#include "core_serialization.hpp"

int main() {
    network test_net({784, 100, 100, 10}, Sigmoid, 0.001);
    test_net.save_current_network_to_file("saved_network{debug}.txt");

    // Load and preprocess CSV...
    // (See full example in main.cpp)

    test_net.online_train(input_data, expected_outputs);
    test_net.save_current_network_to_file("saved_network{debug}.txt");
    return 0;
}


🤝 Contributions

Contributions are welcome! Feel free to submit issues, pull requests, or feature suggestions.

📬 Contact

For any questions or suggestions, feel free to reach out:

📧 Email: k.tym.elsnic@gmail.com

