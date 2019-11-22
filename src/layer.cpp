#include "layer.h"
#include "enums.h"

template<int N, int L>
void Layer<N, L>::randomize() {
    for (Meta<L> conn : connections) {
        conn.randomize();
    }
}

template class Layer<LAYER_1, 0>;
template class Layer<LAYER_2, LAYER_1>;
template class Layer<LAYER_3, LAYER_2>;
