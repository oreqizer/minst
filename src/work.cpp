#include <random>
#include "work.h"
#include "sigmoid.h"
#include "enums.h"

template<int N>
void work::randomize(vector<Connection<N>> &conns) {
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> dist(-EPSILON_INIT, EPSILON_INIT);

    for (auto &c: conns) {
        int i = 0;
        while (i++ < N) {
            c.weights[i] = dist(mt);
        }
    }
}

template void work::randomize(vector<Connection<LAYER_IN_BIAS>> &conns);

template void work::randomize(vector<Connection<LAYER_HIDDEN_BIAS>> &conns);

void work::propagate(vector<float> &neurons, Image &image) {
    int i = 1;
    int size = neurons.size();
    while (i++ < size) {
        neurons[i] = image.pixels[i];
    }
    neurons[0] = 1; // Bias
}

template<int N>
void work::propagate(vector<float> &prevN, vector<Connection<N>> &conns, vector<float> &currN) {
    int i = 0;
    for (auto &c: conns) {
        c.z = 0; // Reset from previous iterations

        int j = 0;
        while (j++ < N) {
            c.z += prevN[j] * c.weights[j];
        }
        currN[i] = sigmoid::classic(c.z);

        i += 1;
    }
}

template void work::propagate(vector<float> &prevN, vector<Connection<LAYER_IN_BIAS>> &conns, vector<float> &currN);

template void work::propagate(vector<float> &prevN, vector<Connection<LAYER_HIDDEN_BIAS>> &conns, vector<float> &currN);
