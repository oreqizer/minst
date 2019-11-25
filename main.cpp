#include <iostream>
#include "src/files.h"
#include "src/network.h"
#include "src/enums.h"

using namespace std;

#define TRAIN_FILE "data/mnist_train.csv", 60000
#define TEST_FILE "data/mnist_test.csv", 10000

int main() {
    cout << "Good morning ladies and gentlemen — buckle your seat belts,";
    cout << "hold tight and get ready for this preworkout-fueled ride of the";
    cout << "ultimate MNIST dataset neural network C++ magnificence!" << endl;
    cout << endl;
    cout << "Info:" << endl;
    cout << "  " << "Mini-batch RMSProp of a 2-layer NN..." << endl;
    cout << "  " << "(" << LAYER_IN << " input neurons) x (" << LAYER_HIDDEN << " hidden neurons) x (" << LAYER_OUT
         << " output neurons)" << endl;
    cout << "  " << "Learning rate: " << RATE << endl;
    cout << "  " << "Gamma: " << RHO << endl;
    cout << "  " << "Epochs: " << EPOCHS << endl;
    cout << "  " << "Batch size: " << BATCH << endl;
    cout << endl;

    Network network;

    // Train
    vector<Image> trains = files::load(TRAIN_FILE);

    network.train(trains);

    // Test
    vector<Image> tests = files::load(TEST_FILE);

    cout << "Testing training data..." << endl;
    network.test(trains);

    cout << "Testing actual data..." << endl;
    network.test(tests);

    return 0;
}
