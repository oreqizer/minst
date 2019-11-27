#include <random>
#include "work.h"
#include "sigmoid.h"
#include "enums.h"

template<int N>
void work::randomize(vector<Connection<N>> &conns) {
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> dist(-WEIGHT_INIT, WEIGHT_INIT);

    for (Connection<N> &c: conns) {
        int i = 0;
        while (i < N) {
            c.weights[i] = dist(mt);

            i += 1;
        }
    }
}

template void work::randomize(vector<Connection<LAYER_IN_BIAS>> &conns);

//template void work::randomize(vector<Connection<LAYER_HIDDEN_1_BIAS>> &conns);

//template void work::randomize(vector<Connection<LAYER_HIDDEN_2_BIAS>> &conns);

void work::propagate(vector<float> &neurons, Image &image) {
    int i = 0;
    int size = neurons.size();
    while (i < size) {
        neurons[i + 1] = image.pixels[i];

        i += 1;
    }
    neurons[0] = BIAS;
}

template<int N>
void work::propagate(vector<float> &prevN, vector<Connection<N>> &conns, vector<float> &currN) {
    int i = 0;
    for (Connection<N> &c: conns) {
        c.z = 0; // Reset from previous iterations

        int j = 0;
        while (j < N) {
            c.z += prevN[j] * c.weights[j];

            j += 1;
        }
        currN[i + 1] = sigmoid::classic(c.z);

        i += 1;
    }
    currN[0] = BIAS;
}

template void work::propagate(vector<float> &prevN, vector<Connection<LAYER_IN_BIAS>> &conns, vector<float> &currN);

//template void
//work::propagate(vector<float> &prevN, vector<Connection<LAYER_HIDDEN_1_BIAS>> &conns, vector<float> &currN);

template<int N>
void work::propagateOut(vector<float> &prevN, vector<Connection<N>> &conns, vector<float> &currN) {
    int i = 0;
    for (Connection<N> &c: conns) {
        c.z = 0; // Reset from previous iterations

        int j = 0;
        while (j < N) {
            c.z += prevN[j] * c.weights[j];

            j += 1;
        }
        currN[i] = sigmoid::classic(c.z);

        i += 1;
    }
}

template void
//work::propagateOut(vector<float> &prevN, vector<Connection<LAYER_HIDDEN_2_BIAS>> &conns, vector<float> &currN);
work::propagateOut(vector<float> &prevN, vector<Connection<LAYER_HIDDEN_1_BIAS>> &conns, vector<float> &currN);

template<int N>
void work::delta(vector<Connection<N>> &conns, vector<float> &neurons, Image &image) {
//    vector<float> target(conns.size());
//
//    target[image.label] = 1;

    int index = 0;
    for (auto &c: conns) {
//        c.delta = sigmoid::prime(c.z) * (neurons[index] - target[index]);
        c.delta = sigmoid::prime(c.z) * (neurons[index] - image.label);

        index += 1;
    }
}

//template void work::delta(vector<Connection<LAYER_HIDDEN_2_BIAS>> &conns, vector<float> &neurons, Image &image);
template void work::delta(vector<Connection<LAYER_HIDDEN_1_BIAS>> &conns, vector<float> &neurons, Image &image);

template<int P, int C>
void work::delta(vector<Connection<P>> &prevC, vector<Connection<C>> &currC) {
    int index = 0;
    for (Connection<C> &c: currC) {
        c.delta = 0;

        for (Connection<P> &p: prevC) {
            c.delta += p.delta * p.weights[index];
        }
        c.delta *= sigmoid::prime(c.z);

        index += 1;
    }
}

//template void
//work::delta(vector<Connection<LAYER_HIDDEN_2_BIAS>> &prevC, vector<Connection<LAYER_HIDDEN_1_BIAS>> &currC);

template void work::delta(vector<Connection<LAYER_HIDDEN_1_BIAS>> &prevC, vector<Connection<LAYER_IN_BIAS>> &currC);

template<int N>
void work::updateGradient(vector<Connection<N>> &conns, vector<float> &neurons) {
    for (Connection<N> &c : conns) {
        int index = 0;
        int size = neurons.size();
        while (index < size) {
            c.gradients[index] += c.delta * neurons[index];

            index += 1;
        }
    }
}

//template void work::updateGradient(vector<Connection<LAYER_HIDDEN_2_BIAS>> &conns, vector<float> &neurons);

template void work::updateGradient(vector<Connection<LAYER_HIDDEN_1_BIAS>> &conns, vector<float> &neurons);

//template void work::updateGradient(vector<Connection<LAYER_IN_BIAS>> &conns, vector<float> &neurons);

template<int N>
void work::updateWeights(float lr, vector<Connection<N>> &conns) {
    for (Connection<N> &c: conns) {
        int index = 0;
        while (index < N) {
//            c.rmsprops[index] = MOMENTUM * c.rmsprops[index] + (1 - MOMENTUM) * pow(c.gradients[index], 2);
//            c.weights[index] -= lr * (c.gradients[index] / BATCH) / (sqrt(c.rmsprops[index]) + EPSILON);
            c.weights[index] -= lr * (c.gradients[index] / BATCH); /// (sqrt(c.rmsprops[index]) + EPSILON);
            c.gradients[index] = 0; // Reset accumulator of batch gradient

            index += 1;
        }
    }
}

//template void work::updateWeights(float lr, vector<Connection<LAYER_HIDDEN_2_BIAS>> &conns);

template void work::updateWeights(float lr, vector<Connection<LAYER_HIDDEN_1_BIAS>> &conns);

//template void work::updateWeights(float lr, vector<Connection<LAYER_IN_BIAS>> &conns);
