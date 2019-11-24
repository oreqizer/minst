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

    void propagate(Image &image);

    void backpropagate(Image &image);

    void updateWeights();

    void train(const vector<reference_wrapper<Image>> &images);

private:
    Layer layer1;
    Layer layer2;
    Layer layer3;
};

#endif //MNIST_NETWORK_H
