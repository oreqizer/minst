#include <random>
#include <iostream>
#include <fstream>
#include "network.h"
#include "work.h"
#include "enums.h"

using namespace std;

Network::Network() :
        neuronsIn(vector<float>(LAYER_IN_BIAS)),
        neuronsHidden1(vector<float>(LAYER_HIDDEN_1_BIAS)),
        neuronsHidden2(vector<float>(LAYER_HIDDEN_2_BIAS)),
        neuronsOut(vector<float>(LAYER_OUT)),
        connectionsHidden1(vector<Connection<LAYER_IN_BIAS>>(LAYER_HIDDEN_1)),
        connectionsHidden2(vector<Connection<LAYER_HIDDEN_1_BIAS>>(LAYER_HIDDEN_2)),
        connectionsOut(vector<Connection<LAYER_HIDDEN_2_BIAS>>(LAYER_OUT)) {}

void Network::propagate(Image &image) {
    work::propagate(neuronsIn, image);
    work::propagate(neuronsIn, connectionsHidden1, neuronsHidden1);
    work::propagate(neuronsHidden1, connectionsHidden2, neuronsHidden2);
    work::propagateOut(neuronsHidden2, connectionsOut, neuronsOut);
}

void Network::backpropagate(Image &image) {
    work::delta(connectionsOut, neuronsOut, image);
    work::delta(connectionsOut, connectionsHidden2);
    work::delta(connectionsHidden2, connectionsHidden1);

    work::updateGradient(connectionsOut, neuronsHidden2);
    work::updateGradient(connectionsHidden2, neuronsHidden1);
    work::updateGradient(connectionsHidden1, neuronsIn);
}

void Network::updateWeights(float lr) {
    work::updateWeights(lr, connectionsOut);
    work::updateWeights(lr, connectionsHidden2);
    work::updateWeights(lr, connectionsHidden1);
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
    for (auto &n : neuronsOut) {
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
    float lr = RATE;
    while (epoch < EPOCHS) {
        vector<Image> inputs = images;

        random_device rd;
        mt19937 dist(rd());
        shuffle(inputs.begin(), inputs.end(), dist);

        int index = 0;
        for (Image &image: inputs) {
            float err = 0;

            propagate(image);
            backpropagate(image);

            err += error(image) / BATCH;

            if (index % BATCH == 0) {
                cout << "Epoch " << epoch + 1 << " / " << EPOCHS
                     << ", batch " << index / BATCH << " / " << size / BATCH
                     << ", LR " << lr
                     << ", loss " << err << '\r' << flush;

                updateWeights(lr);

                lr /= float(1 + DECAY * epoch);

                err = 0;
            }

            index += 1;
        }
        epoch += 1;
    }
}

void Network::test(const vector<Image> &images, const string &out) {
    ofstream file(out);

    int size = images.size();

    int index = 0;
    int correct = 0;
    for (auto image : images) {
        propagate(image);

        int guess = prediction();
        if (guess == image.label) {
            correct += 1;
        }

        cout << "Accuracy: " << 100 * float(correct) / float(index) << "%" << '\r' << flush;
        file << guess << endl;

        index += 1;
    }
    cout << "Accuracy: " << 100 * float(correct) / float(size) << "%" << endl;
    cout << endl;
    cout << "Correct guesses: " << correct << " / " << size << endl;
    cout << endl;

    file.close();
}
