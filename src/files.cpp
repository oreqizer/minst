#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "files.h"

using namespace std;

/**
 * Images are formatted:
 * label,...pixels
 */
istream& operator>>(istream& str, Image& data) {
    string line;
    getline(str, line);
    if (line.empty()) {
        return str;
    }

    stringstream ss(line);
    string cell;

    // Label
    getline(ss, cell, ',');
    data.label = stoi(cell);

    // Pixels
    vector<int> pixels;
    while (getline(ss, cell, ',')) {
        data.pixels.push_back(stoi(cell));
    }
    return str;
}

vector<reference_wrapper<Image>> files::load(const string &filename) {
    ifstream file(filename);

    vector<reference_wrapper<Image>> images;
    Image img;
    while (file >> img) {
        images.emplace_back(img);
    }
    return images;
}
