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
}

#endif //MNIST_WORK_H
