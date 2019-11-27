#include <iostream>
#include "src/files.h"
#include "src/network.h"
#include "src/enums.h"

using namespace std;

#define TRAIN_FILE "data/mnist_train.csv", 60000
#define TEST_FILE "data/mnist_test.csv", 10000

#define TRAIN_OUT "trainPredictions"
#define TEST_OUT "actualTestPredictions"

int main() {
    cout << "Good morning ladies and gentlemen — buckle your seat belts," << endl;
    cout << "hold tight and get ready for this preworkout-fueled ride of the" << endl;
    cout << "ultimate MNIST dataset neural network C++ magnificence!" << endl;
    cout << endl;
    cout << "Info:" << endl;
    cout << "  " << "Mini-batch RMSProp of a 3-layer NN..." << endl;
    cout << "  " << "(" << LAYER_IN << " neurons) x ("
         << LAYER_HIDDEN_1 << " neurons) x ("
         << LAYER_HIDDEN_2 << " neurons) x ("
         << LAYER_OUT << " neurons)" << endl;
    cout << "  " << "Learning rate: " << RATE << endl;
    cout << "  " << "Decay: " << DECAY << endl;
    cout << "  " << "Momentum: " << MOMENTUM << endl;
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
    network.test(trains, TRAIN_OUT);

    cout << "Testing actual data..." << endl;
    network.test(tests, TEST_OUT);

    return 0;
}
