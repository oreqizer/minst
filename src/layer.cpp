#include <cmath>
#include <random>
#include <iostream>
#include "layer.h"
#include "sigmoid.h"
#include "enums.h"

Layer::Layer(int size) : bias() {
    neurons.reserve(size);
    while (size--) {
        Neuron n;

        neurons.emplace_back(n);
    }
}

Layer::Layer(int size, Layer &previous) : bias() {
    vector<Neuron> biased(previous.neurons);

    // Bias is 1st
    biased.emplace_back(previous.bias);

    neurons.reserve(size);
    while (size--) {
        Neuron n(biased);

        neurons.emplace_back(n);
    }
}

void Layer::randomize() {
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> dist(-EPSILON_INIT, EPSILON_INIT);

    for (Neuron &neuron : neurons) {
        for (auto conn : neuron.connections) {
            conn.weight = dist(mt);
        }
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
    for (Neuron &neuron : neurons) {
        neuron.activation = inputs[index++];
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

vector<float> Layer::delta(Image &image) {
    int size = image.pixels.size();

    vector<float> target(size);
    vector<float> delta(size);

    target[image.label] = 1;

    int n = 0;
    for (Neuron &neuron : neurons) {
        delta[n] = sigmoid::prime(neuron.z) * (target[n] - neuron.activation);

        n += 1;
    }
    return delta;
}

vector<float> Layer::delta(const vector<float> &previous) {
    vector<float> delta(neurons.size());

    int n = 0;
    for (Neuron &neuron : neurons) {
        int c = 0;
        for (auto conn : neuron.connections) {
            delta[n] += previous[c++] * conn.weight;
        }
        delta[n++] *= sigmoid::prime(neuron.z);
    }
    return delta;
}

void Layer::updateGradient(const vector<float> &delta) {
    for (Neuron &neuron : neurons) {
        int c = 0;
        for (auto conn : neuron.connections) {
            conn.gradient += delta[c++] * neuron.activation;
        }
    }
}

void Layer::updateWeights() {
    for (Neuron &neuron : neurons) {
        for (auto conn : neuron.connections) {
            conn.rmsprop = RHO * conn.rmsprop + (1 - RHO) * pow(conn.gradient, 2);
            conn.weight += RATE * (conn.gradient / BATCH) / (sqrt(conn.rmsprop) + EPSILON);
            conn.gradient = 0; // Reset accumulator of batch gradient
        }
    }
}
