#ifndef MNIST_NETWORK_H
#define MNIST_NETWORK_H

#include <vector>
#include "image.h"
#include "Connection.h"
#include "enums.h"

using namespace std;

class Network {
public:
    Network();

    ~Network() = default;

    void propagate(Image &image);

    void backpropagate(Image &image);

    void updateWeights(float lr);

    float error(Image &image);

    int prediction();

    void train(const vector<Image> &images);

    void test(const vector<Image> &images);

private:
    vector<float> neuronsIn;
    vector<float> neuronsHidden1;
//    vector<float> neuronsHidden2;
    vector<float> neuronsOut;

    vector<Connection<LAYER_IN_BIAS>> connectionsHidden1;
//    vector<Connection<LAYER_HIDDEN_1_BIAS>> connectionsHidden2;
//    vector<Connection<LAYER_HIDDEN_2_BIAS>> connectionsOut;
    vector<Connection<LAYER_HIDDEN_1_BIAS>> connectionsOut;
};

#endif //MNIST_NETWORK_H
