#include <cmath>
#include "sigmoid.h"

float sigmoid::classic(float z) {
    return 1 / (1 + exp(-z));
}

float sigmoid::prime(float z) {
    auto res = classic(z);

    return res * (1 - res);
}
