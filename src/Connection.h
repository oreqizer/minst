#ifndef MNIST_CONNECTION_H
#define MNIST_CONNECTION_H

#include <vector>

using namespace std;

template<int N>
class Connection {
public:
    Connection() : z(), delta(), weights(vector<float>(N)), gradients(vector<float>(N)), rmsprops(vector<float>(N)) {};

    ~Connection() = default;

    float z;
    float delta;
    vector<float> weights;
    vector<float> gradients;
    vector<float> rmsprops;
};


#endif //MNIST_CONNECTION_H
