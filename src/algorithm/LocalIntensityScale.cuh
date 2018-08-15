#ifndef LOCAL_INTENSITY_SCALE_CUH
#define LOCAL_INTENSITY_SCALE_CUH

#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRParameters.hpp"

template <typename T, typename S>
void localIntensityScaleCuda(const PixelData<T> &image, const APRParameters &par, S *cudaImage, S *cudaTemp, cudaStream_t aStream = 0);


#endif