#ifndef MNIST_NEURON_H
#define MNIST_NEURON_H

#include <vector>
#include "connection.h"

using namespace std;

class Neuron {
public:
    Neuron();

    Neuron(vector<Neuron> &origin, Neuron &bias);

    ~Neuron() = default;

    float activation;
    float z;
    vector<Connection> connections;
    Connection *bias; // Nullable
};

#endif //MNIST_NEURON_H
