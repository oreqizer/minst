#include "input.h"
#include "enums.h"

template<int N>
void Input<N>::randomize() {
    for (auto n : neurons) {
        n.randomize();
    }
}

template<int N>
void Input<N>::clear() {
    z = 0;
}

template<int N>
void Input<N>::updateZ(float input) {
    for (auto n : neurons) {
        z += input * n.weight;
    }
}

template class Input<0>;
template class Input<LAYER_1>;
template class Input<LAYER_2>;
