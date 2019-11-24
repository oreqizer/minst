#ifndef MNIST_CONNECTION_H
#define MNIST_CONNECTION_H

// Forward declaration (https://stackoverflow.com/questions/625799)
class Neuron;

class Connection {
public:
    explicit Connection(Neuron &target): origin(target), weight(), gradientAccumulator(), rmsprop() {}
    ~Connection() = default;

    void randomize();

    Neuron &origin;
    float weight;
    float gradientAccumulator;
    float rmsprop;
};

#endif //MNIST_CONNECTION_H
