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

float Network::error(Image &image) {
    auto size = layerOut.neurons.size();

    vector<float> target(size);

    target[image.label] = 1;

    int index = 0;
    float acc = 0;
    for (const auto &n : layerOut.neurons) {
        acc += float(pow(target[index] - n.activation, 2)) / float(size);
    }
    return acc;
}

int Network::prediction() {
    int res = 0;
    int index = 0;
    float max = -1;
    for (const auto &n : layerOut.neurons) {
        if (n.activation > max) {
            max = n.activation;
            res = index;
        }
        index += 1;
    }
    return res;
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
            float err = 0;
            int iteration = BATCH;
            while (iteration--) {
                // Choose a random image
                Image image = images[dist(mt)];

                propagate(image);
                backpropagate(image);

                err += error(image);
            }
            cout << "Epoch " << epoch << " / " << EPOCHS
                 << ", batch " << batch << " / " << batches
                 << ", loss " << err << '\r' << flush;

            updateWeights();
        }
    }
}

void Network::test(const vector<Image> &images) {
    int size = images.size();

    int index = 0;
    int correct = 0;
    for (auto image : images) {
        propagate(image);

        int guess = prediction();
        if (guess == image.label) {
            correct += 1;
        }

        cout << "Correct guesses: " << correct << " / " << ++index << '\r' << flush;
    }
    cout << "Correct guesses: " << correct << " / " << size << endl;
    cout << endl;
    cout << "Accuracy: " << 100 * float(correct) / float(size) << "%" << endl;
    cout << endl;
}
