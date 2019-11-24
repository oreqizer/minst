#include "neuron.h"

Neuron::Neuron() : activation(), z(), connections() {}

Neuron::Neuron(const vector<Neuron> &origin) : activation(), z() {
    connections.reserve(origin.size());
    for (Neuron n : origin) {
        connections.emplace_back(Connection(n));
    }
}
