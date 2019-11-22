#include <vector>
#include "image.h"

using namespace std;

Image::Image(): label(0), pixels(vector<int>()) {};

Image::~Image() = default;
