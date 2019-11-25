#include "neuron.h"

Neuron::Neuron() : activation(), z(), connections(), bias(nullptr) {}

Neuron::Neuron(vector<Neuron> &origin, Neuron &bias) : activation(), z(), bias(new Connection(bias)) {
    connections.reserve(origin.size());
    for (Neuron &n : origin) {
        connections.emplace_back(Connection(n));
    }
}
