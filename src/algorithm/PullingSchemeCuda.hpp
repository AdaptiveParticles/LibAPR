//
// Created by gonciarz on 10/18/18.
//

#ifndef LIBAPR_PULLINGSCHEMECUDA_HPP
#define LIBAPR_PULLINGSCHEMECUDA_HPP


#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/GenInfo.hpp"
#include "algorithm/ParticleCellTreeCuda.cuh"

using TreeElementType = uint8_t;

template <typename T>
std::vector<PixelData<uint8_t>> computeOvpcCuda(const PixelData<T> &input, const GenInfo &gi);

template <typename ImgType>
void computeOvpcCuda(ImgType *in, ParticleCellTreeCuda &pct, const GenInfo &gi, cudaStream_t stream);

#endif //LIBAPR_PULLINGSCHEMECUDA_HPP
