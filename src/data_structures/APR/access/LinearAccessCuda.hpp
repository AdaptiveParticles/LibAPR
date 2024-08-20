#ifndef APR_LINEARACCESSCUDA_HPP
#define APR_LINEARACCESSCUDA_HPP

#include "algorithm/APRParameters.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/APR/GenInfo.hpp"

typedef struct {
    VectorData<uint16_t> y_vec;
    VectorData<uint64_t> xz_end_vec;
    VectorData<uint64_t> level_xz_vec;
} LinearAccessCudaStructs;

LinearAccessCudaStructs initializeLinearStructureCuda(GenInfo &gi, const APRParameters &apr_parameters, std::vector<PixelData<uint8_t>> &pct);


#endif //APR_LINEARACCESSCUDA_HPP
