#include "meta.h"
#include "enums.h"

template<int N>
void Meta<N>::randomize() {
    for (auto n : neurons) {
        n.randomize();
    }
}

template class Meta<0>;
template class Meta<LAYER_1>;
template class Meta<LAYER_2>;
