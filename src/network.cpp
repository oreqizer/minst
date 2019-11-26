#include <random>
#include <iostream>
#include "network.h"
#include "work.h"
#include "enums.h"

using namespace std;

Network::Network() :
        neuronsIn(vector<float>(LAYER_IN)),
        neuronsHidden1(vector<float>(LAYER_HIDDEN)),
        neuronsHidden2(vector<float>(LAYER_HIDDEN)),
        neuronsOut(vector<float>(LAYER_OUT)),
        connectionsHidden1(vector<Connection<LAYER_IN_BIAS>>(LAYER_HIDDEN)),
        connectionsHidden2(vector<Connection<LAYER_HIDDEN_BIAS>>(LAYER_HIDDEN)),
        connectionsOut(vector<Connection<LAYER_HIDDEN_BIAS>>(LAYER_OUT)) {}

void Network::propagate(Image &image) {
    work::propagate(neuronsIn, image);
    work::propagate(neuronsIn, connectionsHidden1, neuronsHidden1);
    work::propagate(neuronsHidden1, connectionsHidden2, neuronsHidden2);
    work::propagate(neuronsHidden2, connectionsOut, neuronsOut);
}

void Network::backpropagate(Image &image) {
    // TODO
}

void Network::updateWeights() {
    // TODO
}

float Network::error(Image &image) {
    auto size = neuronsOut.size();

    vector<float> target(size);

    target[image.label] = 1;

    int index = 0;
    float acc = 0;
    for (const auto &n : neuronsOut) {
        acc += float(pow(target[index] - n, 2)) / float(size);
    }
    return acc;
}

int Network::prediction() {
    int res = 0;
    int index = 0;
    float max = -1;
    for (const auto &n : neuronsOut) {
        if (n > max) {
            max = n;
            res = index;
        }
        index += 1;
    }
    return res;
}

void Network::train(const vector<Image> &images) {
    work::randomize(connectionsHidden1);
    work::randomize(connectionsHidden2);
    work::randomize(connectionsOut);

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
