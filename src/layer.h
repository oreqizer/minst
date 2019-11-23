#ifndef MNIST_LAYER_H
#define MNIST_LAYER_H

#include <vector>
#include "image.h"
#include "input.h"

using namespace std;

/**
 * One layer in a network
 * @tparam N number of this layer's neurons
 * @tparam L number of previous layer's neurons
 */
template <int N, int L>
class Layer {
public:
    Layer(): activations(vector<float>(N)), bias(0), connections(vector<Input<L>>()) {}
    ~Layer() = default;

    void randomize();
    void propagate(vector<float> inputs);

    vector<float> activations;

private:
    float bias;
    vector<Input<L>> connections;
};

#endif //MNIST_LAYER_H
