#include "neuron.h"

Neuron::Neuron(): activation(), z(), connections() {}

Neuron::Neuron(const vector<reference_wrapper<Neuron>> &origin): activation(), z() {
    connections.reserve(origin.size());
    for (auto n : origin) {
        connections.emplace_back(Connection(n.get()));
    }
}

void Neuron::randomize() {
    for (auto n : connections) {
        n.randomize();
    }
}
