#include <random>
#include "network.h"

using namespace std;

Network::Network():
    layer1(Layer<LAYER_1, 0>()),
    layer2(Layer<LAYER_2, LAYER_1>()),
    layer3(Layer<LAYER_3, LAYER_2>()) {}

void Network::train(const vector<reference_wrapper<Image>> &images) {
    layer2.randomize();
    layer3.randomize();

    auto size = images.size();

    int epochs = EPOCHS;
    while (epochs--) {
        random_device rd;
        mt19937 mt(rd());
        uniform_int_distribution<> dist(0, size - 1);

        auto batches = size / BATCH;
        while (batches--) {
            int batch = BATCH;
            while (batch--) {
                // Choose a random image
                auto image = images[dist(mt)].get();

                layer1.propagate(image.activations());
                layer2.propagate(layer1.activations);
                layer3.propagate(layer2.activations);
                // TODO backpropagate
            }
            // TODO update weights
        }
    }
}
