#ifndef LOCAL_INTENSITY_SCALE_CUH
#define LOCAL_INTENSITY_SCALE_CUH

#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRParameters.hpp"

template <typename S>
void runLocalIntensityScalePipeline(const PixelDataDim &image, const APRParameters &par, S *cudaImage, S *cudaTemp, S *lstPadded, S *lst2Padded, cudaStream_t aStream);

#endif