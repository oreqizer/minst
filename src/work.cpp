#include <random>
#include "work.h"
#include "enums.h"

template<int N>
void randomize(vector<Connection<N>> &conns) {
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
