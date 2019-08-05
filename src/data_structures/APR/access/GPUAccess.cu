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
    ScopedCudaMemHandler<uint16_t*, JUST_ALLOC> y_vec;
    ScopedCudaMemHandler<uint64_t*, JUST_ALLOC> xz_end_vec;
    ScopedCudaMemHandler<uint64_t*, JUST_ALLOC> level_xz_vec;

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
    data->y_vec.copyH2D();
    data->xz_end_vec.copyH2D();
    data->level_xz_vec.copyH2D();
}

void GPUAccess::copy2Device(const size_t numElements){
    data->y_vec.copyH2D(numElements);
    data->xz_end_vec.copyH2D();
    data->level_xz_vec.copyH2D();
}

void GPUAccess::copy2Host(){
    data->y_vec.copyD2H();
    data->xz_end_vec.copyD2H();
    data->level_xz_vec.copyD2H();
}

GPUAccessHelper::GPUAccessHelper(GPUAccess& gpuAccess_,LinearAccess& linearAccess_){
    gpuAccess = &gpuAccess_;
    linearAccess = &linearAccess_;
}

uint16_t* GPUAccessHelper::get_y_vec_ptr(){
    return gpuAccess->data->y_vec.get();
}

uint64_t* GPUAccessHelper::get_level_xz_vec_ptr(){
    return gpuAccess->data->level_xz_vec.get();
}

uint64_t* GPUAccessHelper::get_xz_end_vec_ptr(){
    return gpuAccess->data->xz_end_vec.get();
}



#include "data_structures/APR/particles/ParticleDataGpu.hpp"

template<typename DataType>
template<typename T>
class ParticleDataGpu<DataType>::ParticleDataGpuImpl {

public:
    ScopedCudaMemHandler<T *, H2D> part_data;

    ParticleDataGpuImpl() = default;

    ~ParticleDataGpuImpl() = default;

};

template<typename DataType>
ParticleDataGpu<DataType>::ParticleDataGpu(): data{new ParticleDataGpuImpl<DataType>}
{}

template<typename DataType>
ParticleDataGpu<DataType>::~ParticleDataGpu()
{}

template<typename DataType>
void ParticleDataGpu<DataType>::init(std::vector<DataType>& cpu_data){
    data->part_data.initialize(cpu_data.data(),cpu_data.size());
}

template<typename DataType>
DataType* ParticleDataGpu<DataType>::getGpuData(){
    return data->part_data.get();
}

template<typename DataType>
void ParticleDataGpu<DataType>::sendDataToGpu(){
    data->part_data.copyH2D();
}

template<typename DataType>
void ParticleDataGpu<DataType>::getDataFromGpu(){
    data->part_data.copyD2H();
}

template class ParticleDataGpu<uint16_t>;
template class ParticleDataGpu<uint8_t>;
template class ParticleDataGpu<float>;
template class ParticleDataGpu<double>;
template class ParticleDataGpu<int>;
template class ParticleDataGpu<uint64_t>;
