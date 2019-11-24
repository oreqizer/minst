#include "layer.h"
#include "sigmoid.h"
#include "enums.h"

Layer::Layer(int size) : neurons(vector<reference_wrapper<Neuron>>(size)), bias() {}

Layer::Layer(int size, Layer &previous) : bias() {
    vector<reference_wrapper<Neuron>> ns;
    vector<reference_wrapper<Neuron>> biased(previous.neurons);

    biased.emplace_back(previous.bias);

    ns.reserve(size);
    while (size--) {
        ns.emplace_back(Neuron(biased));
    }
    neurons = ns;
}

void Layer::randomize() {
    for (Neuron conn : neurons) {
        conn.randomize();
    }
}

void Layer::propagate(Image &image) {
    // Reset bias
    bias.activation = 1;

    // Can't swap due to <int> to <float>
    vector<float> inputs(image.pixels.size());
    for (auto pixel : image.pixels) {
        inputs.push_back(pixel);
    }

    int index = 0;
    for (auto input : inputs) {
        Neuron &neuron = neurons[index].get();

        neuron.activation = input;
    }
}

void Layer::propagate() {
    // Reset bias
    bias.activation = 1;

    for (Neuron n : neurons) {
        n.z = 0;
        for (auto conn : n.connections) {
            n.z += conn.origin.activation * conn.weight;
        }
        n.activation = sigmoid::classic(n.z);
    }
}
