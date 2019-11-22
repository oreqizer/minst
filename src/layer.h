#ifndef MNIST_LAYER_H
#define MNIST_LAYER_H

#include <vector>
#include "meta.h"

using namespace std;

/**
 * One layer in a network
 * @tparam N number of this layer's neurons
 * @tparam L number of previous layer's neurons
 */
template <int N, int L>
class Layer {
public:
    Layer(): bias(0), input(vector<float>(N)), connections(vector<Meta<L>>()) {}
    ~Layer() = default;

    void randomize();

private:
    float bias;
    vector<float> input;
    vector<Meta<L>> connections;
};

#endif //MNIST_LAYER_H
