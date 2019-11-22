#include <random>
#include <iostream>
#include "neuron.h"

using namespace std;

void Neuron::randomize() {
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> dist(-EPSILON_INIT, EPSILON_INIT);

    weight = dist(mt);
    cout << weight;
}
