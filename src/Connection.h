#ifndef MNIST_CONNECTION_H
#define MNIST_CONNECTION_H

#include <vector>

using namespace std;

template<int N>
class Connection {
public:
    Connection() : delta(), weights(vector<float>(N)), gradients(vector<float>(N)), rmsprops(vector<float>(N)) {};

    ~Connection() = default;

    float delta;
    vector<float> weights;
    vector<float> gradients;
    vector<float> rmsprops;
};


#endif //MNIST_CONNECTION_H
