#ifndef MNIST_NEURON_H
#define MNIST_NEURON_H

#include <vector>
#include "connection.h"

using namespace std;

class Neuron {
public:
    Neuron();

    explicit Neuron(const vector<Neuron> &origin);

    ~Neuron() = default;

    float activation;
    float z;
    vector<Connection> connections;
};

#endif //MNIST_NEURON_H
