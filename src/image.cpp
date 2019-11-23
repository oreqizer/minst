#include "image.h"

vector<float> Image::activations() {
    vector<float> floats(pixels.size());
    for (auto pixel : pixels) {
        floats.push_back(pixel);
    }
    return floats;
}
