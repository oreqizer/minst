#include "layer.h"
#include "sigmoid.h"
#include "enums.h"

template<int N, int L>
void Layer<N, L>::randomize() {
    for (Input<L> conn : connections) {
        conn.randomize();
    }
}

template<int N, int L>
void Layer<N, L>::propagate(vector<float> inputs) {
    // Reset bias
    bias = 1;

    // 1st layer just copy inputs
    if (connections.empty()) {
        activations.swap(inputs);
        return;
    }

    // Propagate from previous layer
    int index = 0;
    for (Input<L> conn : connections) {
        // Clear from previous iterations
        conn.clear();

        for (auto input : inputs) {
            conn.updateZ(input);
        }
        activations[index++] = sigmoid::classic(conn.z);
    }
}

template class Layer<LAYER_1, 0>;
template class Layer<LAYER_2, LAYER_1>;
template class Layer<LAYER_3, LAYER_2>;
