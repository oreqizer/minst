#include <random>
#include "network.h"

using namespace std;

Network::Network():
    layer1(Layer(LAYER_1)),
    layer2(Layer(LAYER_2, layer1)),
    layer3(Layer(LAYER_3, layer2)) {}

void Network::propagate(Image &image) {
    layer1.propagate(image);
    layer2.propagate();
    layer3.propagate();
}

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

                propagate(image);
                auto error = backpropagate(image);
            }
            // TODO update weights
        }
    }
}
