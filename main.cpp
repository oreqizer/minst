#include <iostream>
#include "src/files.h"
#include "src/network.h"
#include "src/enums.h"

using namespace std;

#define TRAIN_FILE "data/mnist_train.csv", 60000
#define TEST_FILE "data/mnist_test.csv", 10000

int main() {
    cout << "Good morning ladies and gentlemen — buckle your seat belts, hold tight and get ready ";
    cout << "for the ride of this ultimate MNIST dataset neural network C++ magnificence!" << endl;
    cout << endl;
    cout << "Info:" << endl;
    cout << "  " << "Mini-batch RMSProp of a 2-layer NN..." << endl;
    cout << "  " << "(" << LAYER_1 << " input neurons) x (" << LAYER_2 << " hidden neurons) x (" << LAYER_3 << " output neurons)" << endl;
    cout << "  " << "Learning rate: " << RATE << endl;
    cout << "  " << "Gamma: " << RHO << endl;
    cout << "  " << "Epochs: " << EPOCHS << endl;
    cout << "  " << "Batch size: " << BATCH << endl;

    Network network;

    // Train
    vector<Image> trains = files::load(TRAIN_FILE);

    network.train(trains);

    // Test
    vector<Image> tests = files::load(TEST_FILE);

    return 0;
}
