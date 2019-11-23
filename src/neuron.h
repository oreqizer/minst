#ifndef MNIST_NEURON_H
#define MNIST_NEURON_H

#define EPSILON_INIT 0.12     // Random init bounds interval -EPSILON_INIT..EPSILON_INIT

class Neuron {
public:
    Neuron() = default;
    ~Neuron() = default;

    void randomize();

    float weight;
private:
    float gradientAccumulator;
    float rmsprop;
};

#endif //MNIST_NEURON_H
