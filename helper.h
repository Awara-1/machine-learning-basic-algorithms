#ifndef HELPER_H
#define HELPER_H

#endif // HELPER_H

#include <iostream>

float sigmoid(float cell){
    return 1/(1+ std::exp(-cell));
}


