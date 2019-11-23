#ifndef MNIST_INPUT_H
#define MNIST_INPUT_H

#include <vector>
#include "neuron.h"

using namespace std;

/**
 * Contains metadata about the left-connection of a neuron.
 * @tparam N number of connected neurons from the left
 */
template<int N>
class Input {
public:
    explicit Input(): z(0), bias(), neurons(vector<Neuron>(N)) {}
    ~Input() = default;

    void randomize();
    void clear();
    void updateZ(float input);

    float z;

private:
    Neuron bias;
    vector<Neuron> neurons;
};

#endif //MNIST_INPUT_H
