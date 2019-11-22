#include "network.h"

Network::Network():
    layer1(Layer<LAYER_1, 0>()),
    layer2(Layer<LAYER_2, LAYER_1>()),
    layer3(Layer<LAYER_3, LAYER_2>()) {}

void Network::randomize() {
    layer2.randomize();
    layer3.randomize();
}
