#ifndef MNIST_LAYER_H
#define MNIST_LAYER_H

#include <vector>
#include "image.h"
#include "neuron.h"

using namespace std;

/**
 * One layer in a network
 * @tparam N number of this layer's neurons
 * @tparam C number of previous layer's neurons
 */
class Layer {
public:
    explicit Layer(int size);

    explicit Layer(int size, Layer &previous);

    ~Layer() = default;

    void randomize();

    void propagate(Image &image);

    void propagate();

    vector<reference_wrapper<Neuron>> neurons;
    Neuron bias;
};

#endif //MNIST_LAYER_H
