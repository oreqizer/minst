#ifndef MNIST_WORK_H
#define MNIST_WORK_H

#include <vector>
#include "Connection.h"
#include "image.h"

using namespace std;

namespace work {
    template<int N>
    void randomize(vector<Connection<N>> &conns);

    void propagate(vector<float> &neurons, Image &image);

    template<int N>
    void propagate(vector<float> &prevN, vector<Connection<N>> &conns, vector<float> &currN);

    template<int N>
    void propagateOut(vector<float> &prevN, vector<Connection<N>> &conns, vector<float> &currN);

    template<int N>
    void delta(vector<Connection<N>> &conns, vector<float> &neurons, Image &image);

    template<int P, int C>
    void delta(vector<Connection<P>> &prevC, vector<Connection<C>> &currC);

    template<int N>
    void updateGradient(vector<Connection<N>> &conns, vector<float> &neurons);

    template<int N>
    void updateWeights(float lr, vector<Connection<N>> &conns);
}

#endif //MNIST_WORK_H
