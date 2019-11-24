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
Image readRow(const string &line) {
    Image img;

    stringstream ss(line);
    string cell;

    // Label
    getline(ss, cell, ',');
    img.label = stoi(cell);

    // Pixels
    vector<int> pixels;
    while (getline(ss, cell, ',')) {
        img.pixels.push_back(stoi(cell));
    }
    return img;
}

vector<Image> files::load(const string &filename, int size) {
    ifstream file(filename);
    vector<Image> images;

    images.reserve(size);

    string str;
    while (getline(file, str)) {
        Image img = readRow(str);

        images.push_back(img);
    }

    return images;
}
