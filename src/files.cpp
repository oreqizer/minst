#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "files.h"

using namespace std;

const string TRAIN_FILE = "data/mnist_train.csv";
const string TEST_FILE = "data/mnist_test.csv";

/**
 * Images are formatted:
 * label,...pixels
 */
istream& operator>>(istream& str, Image& data) {
    string line;
    getline(str, line);

    stringstream lineStream(line);
    string cell;

    // Label
    getline(lineStream, cell, ',');
    data.label = stoi(cell);

    // Pixels
    vector<int> pixels;
    while (getline(lineStream, cell, ',')) {
        data.pixels.push_back(stoi(cell));
    }
    return str;
}

vector<Image&> files::load(const string &filename) {
    ifstream file(filename);

    vector<Image&> images;
    Image img;
    while (file >> img) {
        images.push_back(img);
    }
    return images;
}
