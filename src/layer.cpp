#include <cmath>
#include <random>
#include "layer.h"
#include "sigmoid.h"
#include "enums.h"

Layer::Layer(int size) {
    neurons.reserve(size);
    while (size--) {
        Neuron n;

        neurons.emplace_back(n);
    }
}

Layer::Layer(int size, Layer &previous) {
    neurons.reserve(size);
    while (size--) {
        Neuron n(previous.neurons, previous.bias);

        neurons.emplace_back(n);
    }
}

void Layer::randomize() {
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> dist(-EPSILON_INIT, EPSILON_INIT);

    for (Neuron &n : neurons) {
        for (auto &conn : n.connections) {
            conn.weight = dist(mt);
        }
        if (n.bias != nullptr) {
            n.bias->weight = dist(mt);
        }
    }
}

void Layer::propagate(Image &image) {
    // Can't swap due to <int> to <float>
    vector<float> inputs;

    inputs.reserve(image.pixels.size());
    for (float pixel : image.pixels) {
        inputs.push_back(pixel);
    }

    int index = 0;
    for (Neuron &n : neurons) {
        n.activation = inputs[index++] / 255;
    }

    bias.activation = 1;
}

void Layer::propagate() {
    for (Neuron &n : neurons) {
        n.z = 0;
        for (auto conn : n.connections) {
            n.z += conn.origin.activation * conn.weight;
        }
        if (n.bias != nullptr) {
            n.z += n.bias->origin.activation * n.bias->weight;
        }
        n.activation = sigmoid::classic(n.z);
    }

    bias.activation = 1;
}

void Layer::delta(Image &image) {
    vector<float> target(neurons.size());

    target[image.label] = 1;

    int index = 0;
    for (Neuron &n : neurons) {
        n.delta = sigmoid::prime(n.activation) * (target[index] - n.activation);

        index += 1;
    }
}

void Layer::delta(Layer &previous) {
    int index = 0;
    for (Neuron &n : neurons) {
        int c = 0;
        for (auto &prev : previous.neurons) {
            n.delta += prev.delta * prev.connections[c++].weight;
        }
        n.delta *= sigmoid::prime(n.activation);

        index += 1;
    }
}

void Layer::updateGradient() {
    int index = 0;
    for (Neuron &n : neurons) {
        for (auto &conn : n.connections) {
            conn.gradient += n.delta * conn.origin.activation;
        }
        if (n.bias != nullptr) {
            n.bias->gradient += n.delta;
        }

        n.delta = 0; // TODO ??
        index += 1;
    }
}

void Layer::updateWeights() {
    for (Neuron &n : neurons) {
        for (auto &conn : n.connections) {
            conn.rmsprop = RHO * conn.rmsprop + (1 - RHO) * pow(conn.gradient, 2);
            conn.weight += RATE * (conn.gradient / BATCH) / (sqrt(conn.rmsprop) + EPSILON);
            conn.gradient = 0; // Reset accumulator of batch gradient
        }
        if (n.bias != nullptr) {
            auto &conn = *n.bias;

            conn.rmsprop = RHO * conn.rmsprop + (1 - RHO) * pow(conn.gradient, 2);
            conn.weight += RATE * (conn.gradient / BATCH) / (sqrt(conn.rmsprop) + EPSILON);
            conn.gradient = 0; // Reset accumulator of batch gradient
        }
    }
}
