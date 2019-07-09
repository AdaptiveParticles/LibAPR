//
// Created by cheesema on 2019-07-09.
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"
#include <chrono>
#include <cstdint>

#include "GPUAccess.hpp"

class GPUAccess::GPUAccessImpl{
public:
    ScopedCudaMemHandler<uint16_t*, H2D> y_vec;
    ScopedCudaMemHandler<uint64_t*, H2D> xz_end_vec;
    ScopedCudaMemHandler<uint64_t*, H2D> level_xz_vec;

    GPUAccessImpl()=default;
    ~GPUAccessImpl()=default;
};

GPUAccess::~GPUAccess() = default;
GPUAccess::GPUAccess(): data{new GPUAccessImpl}{

}
GPUAccess::GPUAccess(GPUAccess&&) = default;

void GPUAccess::init_y_vec(std::vector<uint16_t> &y_vec_) {
    data->y_vec.initialize(y_vec_.data(),y_vec_.size());
}

void GPUAccess::init_xz_end_vec(std::vector<uint64_t>& xz_end_vec){
    data->xz_end_vec.initialize(xz_end_vec.data(),xz_end_vec.size());
}
void GPUAccess::init_level_xz_vec(std::vector<uint64_t>& level_xz_vec){
    data->level_xz_vec.initialize(level_xz_vec.data(),level_xz_vec.size());
}

void GPUAccess::copy2Device(){

}
void GPUAccess::copy2Host(){

}


