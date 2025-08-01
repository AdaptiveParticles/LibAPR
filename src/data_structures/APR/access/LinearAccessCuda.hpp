#ifndef APR_LINEARACCESSCUDA_HPP
#define APR_LINEARACCESSCUDA_HPP

#include "algorithm/APRParameters.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/GenInfo.hpp"
#include "algorithm/ParticleCellTreeCuda.cuh"

typedef struct {
    VectorData<uint16_t> y_vec;
    VectorData<uint64_t> xz_end_vec;
    VectorData<uint64_t> level_xz_vec;
} LinearAccessCudaStructs;

#include "data_structures/APR/access/GenInfoGpuAccess.cuh"

// This is for testing purposes only
LinearAccessCudaStructs initializeLinearStructureCuda(GenInfo &gi, const APRParameters &apr_parameters, std::vector<PixelData<uint8_t>> &pct);

void computeLinearStructureCuda(uint16_t *y_vec_cuda, uint64_t *xz_end_vec_cuda, const uint64_t *level_xz_vec_cuda, ParticleCellTreeCuda &p_map, GenInfo &gi, GenInfoGpuAccess &giga, const APRParameters &apr_parameters, uint64_t counter_total, cudaStream_t aStream);


#endif //APR_LINEARACCESSCUDA_HPP
