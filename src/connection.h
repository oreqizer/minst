#ifndef MNIST_CONNECTION_H
#define MNIST_CONNECTION_H

// Forward declaration (https://stackoverflow.com/questions/625799)
class Neuron;

class Connection {
public:
    explicit Connection(Neuron &target): origin(target), weight(), gradient(), rmsprop() {}
    ~Connection() = default;

    void randomize();

    Neuron &origin;
    float weight;
    float gradient;
    float rmsprop;
};

#endif //MNIST_CONNECTION_H
