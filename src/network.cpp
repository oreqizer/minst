#include <random>
#include <iostream>
#include "network.h"
#include "enums.h"

using namespace std;

Network::Network() :
        layerIn(Layer(LAYER_IN)),
        layerHidden1(Layer(LAYER_HIDDEN, layerIn)),
        layerHidden2(Layer(LAYER_HIDDEN, layerHidden1)),
        layerOut(Layer(LAYER_OUT, layerHidden2)) {}

void Network::propagate(Image &image) {
    layerIn.propagate(image);
    layerHidden1.propagate();
    layerHidden2.propagate();
    layerOut.propagate();
}

void Network::backpropagate(Image &image) {
    layerOut.delta(image);
    layerHidden2.delta(layerOut);
    layerHidden1.delta(layerHidden2);

    // TODO fix these delta functions
    layerOut.updateGradient();
    layerHidden2.updateGradient();
    layerHidden1.updateGradient();
}

void Network::updateWeights() {
    layerOut.updateWeights();
    layerHidden2.updateWeights();
    layerHidden1.updateWeights();
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
    layerHidden1.randomize();
    layerHidden2.randomize();
    layerOut.randomize();

    int size = images.size();

    int epoch = 0;
    while (epoch++ < EPOCHS) {
        random_device rd;
        mt19937 mt(rd());
        uniform_int_distribution<> dist(0, size - 1);

        int batches = size / BATCH;
//        int batches = 2;
        int batch = 0;
        while (batch++ < batches) {
            float err = 0;
            int iteration = BATCH;
//            int iteration = 5;
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
