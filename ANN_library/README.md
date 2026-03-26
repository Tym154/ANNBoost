This library provides a `network` class for a fully connected neural network with learning capabilities. It is header‑only.

1.  Including Headers
    ----------------
    #include "core_network.hpp"
    - for using the core functions of the library

    #include "core_serialization.hpp"
    - for using `load_network_from_file`


2.  Creating the Network
    ----------------
    network net(
        std::vector<int> layer_sizes,   // number of neurons in each layer
        ActivationType hidden_activation,   // activation function for hidden layers
        ActivationType output_activation,   // activation function for the output layer
        double learning_rate,               // initial learning rate
        std::pair<float,float> bias_range   // range for bias initialisation
    );

    Possible values for `ActivationType`:
        Sigmoid, Tanh, ReLU, LeakyReLU, Softmax

    Example:
        // ({input_size, {hidden_layers}, output_size}, hidden_Activation, output_activation, learning_rate, bias_init_range)
        network net({784, 128, 64, 10}, LeakyReLU, Sigmoid, 0.001, {-0.1f, 0.1f});


3.  Training
    ---------
    The `online_train` method performs one forward‑backward iteration for each sample in the given list.
        void online_train(
            const std::vector<std::vector<double>>& inputs,
            const std::vector<std::vector<double>>& expected
        );

    - `inputs`  : vector of input vectors (each vector is one training example).
    - `expected`: vector of target outputs (usually one‑hot encoding).


4.  Evaluation (Forward Pass)
    ------------------------------
    double network_forward_propagation(std::vector<double>& input);
        // Performs a forward pass, returns an error (or 0) – an implementation detail.
        // The prediction can then be obtained with:
    int get_the_most_active_in_output();
        // Returns the index of the neuron with the highest activity in the output layer.

    Example:
        net.network_forward_propagation(input);
        int prediction = net.get_the_most_active_in_output();


5.  Saving and Loading the Network
    --------------------------
    Saving:
        void save_current_network_to_file(const std::string& filename);

    Loading (returns a `network` object):
        network load_network_from_file(const std::string& filename);

    Example:
        net.save_current_network_to_file("my_network.txt");
        network loaded = load_network_from_file("my_network.txt");


6.  Learning Rate
    ----------------
    The `learning_rate` attribute is public and can be changed during training:
        net.learning_rate = 0.0005;


7.  Network Structure
    ----------------
    The layers are stored in `std::vector<Layer> layers`. Each layer contains:
        std::vector<Node> nodes_in_layer;   // neurons of the layer
    Each neuron has `weights` and `bias`.

    For debugging, you can print, for example, the number of neurons:
        std::cout << net.layers[0].nodes_in_layer.size();   // input layer


8.  Compilation
    ----------
    In the `ANN_lib` directory:
        make SRC_FILE=your_file.cpp

    The resulting binary will be placed in `bin/`.

    In the `ANN_lib` directory:
        make clean
    
    To delete the compiled library files.


9.  Usage with VSCode
    ----------
    Update the include paths with:
        /ANN_lib/CORE_ANNBoost/include