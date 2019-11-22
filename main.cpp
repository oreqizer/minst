#include "src/files.h"
#include "src/network.h"

#define TRAIN_FILE "data/mnist_train.csv"
#define TEST_FILE "data/mnist_test.csv"

int main() {
    // Train
    auto trains = files::load(TRAIN_FILE);
    Network network;

    network.randomize();

    // Test
    auto tests = files::load(TEST_FILE);

    return 0;
}
