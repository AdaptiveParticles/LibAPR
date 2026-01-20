#ifndef APR_BSPLINEPARAMS_H
#define APR_BSPLINEPARAMS_H


#include <cstddef>


struct BsplineParamsCuda {
    float *bc1;
    float *bc2;
    float *bc3;
    float *bc4;
    size_t k0;
    float b1;
    float b2;
    float norm_factor;
};

#endif //APR_BSPLINEPARAMS_H
