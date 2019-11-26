#include <iostream>
#include "src/files.h"
#include "src/network.h"
#include "src/enums.h"

using namespace std;

#define TRAIN_FILE "data/mnist_train.csv", 60000
#define TEST_FILE "data/mnist_test.csv", 10000

int main() {
    cout << "Good morning ladies and gentlemen — buckle your seat belts," << endl;
    cout << "hold tight and get ready for this preworkout-fueled ride of the" << endl;
    cout << "ultimate MNIST dataset neural network C++ magnificence!" << endl;
    cout << endl;
    cout << "Info:" << endl;
    cout << "  " << "Mini-batch RMSProp of a 3-layer NN..." << endl;
    cout << "  " << "(" << LAYER_IN << " neurons) x ("
         << LAYER_HIDDEN_1 << " neurons) x ("
//         << LAYER_HIDDEN_2 << " neurons) x ("
         << LAYER_OUT << " neurons)" << endl;
    cout << "  " << "Learning rate: " << RATE << endl;
    cout << "  " << "Decay: " << DECAY << endl;
    cout << "  " << "Momentum: " << MOMENTUM << endl;
    cout << "  " << "Epochs: " << EPOCHS << endl;
    cout << "  " << "Batch size: " << BATCH << endl;
    cout << endl;

    Network network;

    // Train
//    vector<Image> trains = files::load(TRAIN_FILE);
//    vector<Image> trains = files::load(TEST_FILE);
    vector<int> p1;
    p1.push_back(0);
    p1.push_back(1);
    vector<int> p2;
    p2.push_back(1);
    p2.push_back(0);
    vector<int> p3;
    p3.push_back(0);
    p3.push_back(0);
    vector<int> p4;
    p4.push_back(1);
    p4.push_back(1);

    vector<Image> trains;
    Image img1;
    img1.label = 1;
    img1.pixels = p1;
    trains.push_back(img1);
    Image img2;
    img2.label = 1;
    img2.pixels = p2;
    trains.push_back(img2);
    Image img3;
    img3.label = 0;
    img3.pixels = p3;
    trains.push_back(img3);
    Image img4;
    img4.label = 0;
    img4.pixels = p4;
    trains.push_back(img4);

    network.train(trains);

    // Test
//    vector<Image> tests = files::load(TEST_FILE);

    cout << "Testing training data..." << endl;
    network.test(trains);

//    cout << "Testing actual data..." << endl;
//    network.test(tests);

    return 0;
}
