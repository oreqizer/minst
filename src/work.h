#ifndef MNIST_WORK_H
#define MNIST_WORK_H

#include <vector>
#include "Connection.h"

using namespace std;

namespace work {
    template<int N>
    void randomize(vector<Connection<N>> &conns);
}

#endif //MNIST_WORK_H
