#ifndef MNIST_IMAGE_H
#define MNIST_IMAGE_H

#include <vector>

using namespace std;

class Image {
public:
    Image(): label(0), pixels(vector<int>()) {}
    ~Image() = default;

    vector<float> activations();

    int label; // 0-9
    vector<int> pixels; // 0-255, length 784 (28x28)
};

#endif //MNIST_IMAGE_H
