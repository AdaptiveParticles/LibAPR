//
// Created by Krzysztof Gonciarz on 8/1/18.
//

#ifndef LIBAPR_COMPUTEPULLINGSCHEMECUDA_H
#define LIBAPR_COMPUTEPULLINGSCHEMECUDA_H

#include "data_structures/Mesh/PixelData.hpp"

template <typename ImageType>
void computeLevelsCuda(const PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, int maxLevel, float relError, float dx = 1, float dy = 1, float dz = 1);

template <typename ImageType>
void gradDivLocalIntensityScale(const ImageType *grad, float *lis, size_t len, float mult_const);

#endif //LIBAPR_COMPUTEPULLINGSCHEMECUDA_HPP
