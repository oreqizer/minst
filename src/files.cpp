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
    cout << "Loading file '" << filename << "'" << endl;

    string str;
    int i = 0;
    while (getline(file, str)) {
        Image img = readRow(str);
        if (i % 250 == 0) {
            cout << "Loaded " << i << " images" << '\r' << flush;
        }

        images.push_back(img);

        i += 1;
    }

    cout << "File loaded!" << endl;
    cout << endl;

    return images;
}
