#include <random>
#include <iostream>
#include "network.h"
#include "enums.h"

using namespace std;

Network::Network() :
        layerIn(Layer(LAYER_IN)),
        layerHidden(Layer(LAYER_HIDDEN, layerIn)),
        layerOut(Layer(LAYER_OUT, layerHidden)) {}

void Network::propagate(Image &image) {
    layerIn.propagate(image);
    layerHidden.propagate();
    layerOut.propagate();
}

void Network::backpropagate(Image &image) {
    auto deltaOut = layerOut.delta(image);
    auto deltaHidden = layerHidden.delta(deltaOut);

    layerOut.updateGradient(deltaOut);
    layerHidden.updateGradient(deltaHidden);
}

void Network::updateWeights() {
    layerOut.updateWeights();
    layerHidden.updateWeights();
}

void Network::train(const vector<Image> &images) {
    layerHidden.randomize();
    layerOut.randomize();

    int size = images.size();

    int epoch = 0;
    while (epoch++ < EPOCHS) {
        random_device rd;
        mt19937 mt(rd());
        uniform_int_distribution<> dist(0, size - 1);

        int batches = size / BATCH;
        int batch = 0;
        while (batch++ < batches) {
            cout << "Epoch " << epoch << " / " << EPOCHS << ", batch " << batch << " / " << batches << endl;

            int iteration = BATCH;
            while (iteration--) {
                // Choose a random image
                Image image = images[dist(mt)];

                propagate(image);
                backpropagate(image);
            }
            updateWeights();
        }
    }
}
