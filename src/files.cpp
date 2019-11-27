#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include "files.h"

using namespace std;

int readLabelsRow(istream &ss) {
    string cell;
    getline(ss, cell);

    // Label
    return stoi(cell);
}

vector<int> readVectorsRow(istream &ss) {
    string line;
    getline(ss, line);

    stringstream cells(line);
    string cell;

    // Pixels
    vector<int> pixels;
    while (getline(cells, cell, ',')) {
        pixels.push_back(stoi(cell));
    }
    return pixels;
}

vector<Image> files::load(const string &labelFile, const string &vectorFile, int size) {
    ifstream labels(labelFile);
    ifstream vectors(vectorFile);
    vector<Image> images;

    images.reserve(size);

    int i = 0;
    while (i < size) {
        Image img;
        img.label = readLabelsRow(labels);
        img.pixels = readVectorsRow(vectors);

        if (i % 250 == 0) {
            cout << "Loaded " << i << " images" << '\r' << flush;
        }

        images.push_back(img);

        i += 1;
    }

    labels.close();
    vectors.close();

    cout << "File loaded!" << endl;
    cout << endl;

    return images;
}
