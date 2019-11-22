#ifndef MNIST_META_H
#define MNIST_META_H

#include <vector>
#include "neuron.h"

using namespace std;

/**
 * Contains metadata about the left-connection of a neuron.
 * @tparam N number of connected neurons from the left
 */
template<int N>
class Meta {
public:
    explicit Meta(): z(0), bias(), neurons(vector<Neuron>(N)) {}
    ~Meta() = default;

    void randomize();

private:
    float z;
    Neuron bias;
    vector<Neuron> neurons;
};

#endif //MNIST_META_H
