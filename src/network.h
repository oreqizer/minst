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

    float error(Image &image);

    int prediction();

    void train(const vector<Image> &images);

    void test(const vector<Image> &images);

private:
    Layer layerIn;
    Layer layerHidden;
    Layer layerOut;
};

#endif //MNIST_NETWORK_H
