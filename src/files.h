#ifndef TEXTURES_FILES_H
#define TEXTURES_FILES_H

#include <vector>
#include <string>
#include "image.h"

using namespace std;

namespace files {
    vector<reference_wrapper<Image>> load(const string &filename);
}

istream& operator>>(istream& str, Image& data);

#endif //TEXTURES_FILES_H
