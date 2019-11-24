#ifndef TEXTURES_FILES_H
#define TEXTURES_FILES_H

#include <vector>
#include <string>
#include "image.h"

using namespace std;

namespace files {
    vector<Image> load(const string &filename, int size);
}

#endif //TEXTURES_FILES_H
