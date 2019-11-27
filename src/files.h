#ifndef TEXTURES_FILES_H
#define TEXTURES_FILES_H

#include <vector>
#include <string>
#include "image.h"

using namespace std;

namespace files {
    vector<Image> load(const string &labelFile, const string &vectorFile, int size);
}

#endif //TEXTURES_FILES_H
