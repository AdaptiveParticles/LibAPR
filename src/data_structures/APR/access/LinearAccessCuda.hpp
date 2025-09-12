#ifndef APR_LINEARACCESSCUDA_HPP
#define APR_LINEARACCESSCUDA_HPP

#include "algorithm/APRParameters.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/GenInfo.hpp"
#include "algorithm/ParticleCellTreeCuda.cuh"

template <typename ImgType>
struct LinearAccessCudaStructs {
    VectorData<uint16_t> y_vec;
    VectorData<uint64_t> xz_end_vec;
    VectorData<uint64_t> level_xz_vec;

    // temporarily added
    VectorData<ImgType> parts;
};

// explicit instantiation of handled types
template class LinearAccessCudaStructs<uint8_t>;
template class LinearAccessCudaStructs<int>;
template class LinearAccessCudaStructs<uint16_t>;
template class LinearAccessCudaStructs<float>;

#include "data_structures/APR/access/GenInfoGpuAccess.cuh"

// This is for testing purposes only
template <typename ImgType>
LinearAccessCudaStructs<ImgType> initializeLinearStructureCuda(GenInfo &gi, const APRParameters &apr_parameters, std::vector<PixelData<uint8_t>> &pct);

void computeLinearStructureCuda(uint16_t *y_vec_cuda, uint64_t *xz_end_vec_cuda, const uint64_t *level_xz_vec_cuda, ParticleCellTreeCuda &p_map, GenInfo &gi, GenInfoGpuAccess &giga, const APRParameters &apr_parameters, uint64_t counter_total, cudaStream_t aStream);


#endif //APR_LINEARACCESSCUDA_HPP
