#include "neuron.h"

Neuron::Neuron(): activation(), z(), connections(vector<Connection>(0)) {}

Neuron::Neuron(const vector<reference_wrapper<Neuron>> &origin): activation(), z() {
    vector<Connection> conns;

    conns.reserve(origin.size() + 1);
    for (auto n : origin) {
        conns.emplace_back(Connection(n.get()));
    }
    connections = conns;
}

void Neuron::randomize() {
    for (auto n : connections) {
        n.randomize();
    }
}
