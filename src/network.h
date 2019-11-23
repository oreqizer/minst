#ifndef MNIST_NETWORK_H
#define MNIST_NETWORK_H

#include <vector>
#include "layer.h"
#include "image.h"
#include "enums.h"

using namespace std;

class Network {
public:
    Network();
    ~Network() = default;

    void train(const vector<reference_wrapper<Image>> &images);

private:
    Layer<LAYER_1, 0> layer1;
    Layer<LAYER_2, LAYER_1> layer2;
    Layer<LAYER_3, LAYER_2> layer3;
};

#endif //MNIST_NETWORK_H
