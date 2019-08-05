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

__global__ void fill_y_vec_max_level(const uint64_t* level_xz_vec,
                                     const uint64_t* xz_end_vec,
                                     uint16_t* y_vec,
                                     const uint64_t* level_xz_vec_tree,
                                     const uint64_t* xz_end_vec_tree,
                                     const uint16_t* y_vec_tree,
                                     const int z_num,
                                     const int x_num,
                                     const uint16_t y_num,
                                     const int x_num_parent,
                                     const int level_max_tree);

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

void GPUAccess::copy2Device(const size_t numElements, GPUAccess* tree_access){
    data->y_vec.copyH2D(numElements);
    data->xz_end_vec.copyH2D();
    data->level_xz_vec.copyH2D();

    const int level = tree_access->level_max();
    dim3 blocks_l(tree_access->x_num(level), 1, tree_access->z_num(level));
    dim3 threads_l(32, 2, 2);

    fill_y_vec_max_level<<<blocks_l, threads_l>>>(data->level_xz_vec.get(),
                                                  data->xz_end_vec.get(),
                                                  data->y_vec.get(),
                                                  tree_access->data->level_xz_vec.get(),
                                                  tree_access->data->xz_end_vec.get(),
                                                  tree_access->data->y_vec.get(),
                                                  z_num(level+1),
                                                  x_num(level+1),
                                                  y_num(level+1),
                                                  tree_access->x_num(level),
                                                  level);

    cudaDeviceSynchronize();
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

__global__ void fill_y_vec_max_level(const uint64_t* level_xz_vec,
                                     const uint64_t* xz_end_vec,
                                     uint16_t* y_vec,
                                     const uint64_t* level_xz_vec_tree,
                                     const uint64_t* xz_end_vec_tree,
                                     const uint16_t* y_vec_tree,
                                     const int z_num,
                                     const int x_num,
                                     const uint16_t y_num,
                                     const int x_num_parent,
                                     const int level_max_tree) {



    const int x = 2 * blockIdx.x + threadIdx.y;
    const int z = 2 * blockIdx.z + threadIdx.z;

    if( (x >= x_num) || (z >= z_num) ) {
        return;
    }

    const int y_off = threadIdx.x % 2;
    const int local_id = threadIdx.x / 2;

    size_t xz_start = level_xz_vec_tree[level_max_tree] + blockIdx.x + blockIdx.z * x_num_parent;
    const size_t global_index_begin_p = xz_end_vec_tree[xz_start-1];
    const size_t global_index_end_p = xz_end_vec_tree[xz_start];

    if( (global_index_begin_p == global_index_end_p) ) {
        return;
    }

    xz_start = level_xz_vec[level_max_tree + 1] + x + z * x_num;
    const size_t global_index_begin_0 = xz_end_vec[xz_start-1];
    const size_t global_index_end_0 = xz_end_vec[xz_start];

    const int number_chunks = ((int)(global_index_end_p - global_index_begin_p) + 15) / 16;
    uint16_t y;

    for(int chunk = 0; chunk < number_chunks; ++chunk) {

        __syncthreads();
        if( ((global_index_begin_p + chunk*16 + local_id) < global_index_end_p) ) {
            y = 2 * y_vec_tree[global_index_begin_p + chunk*16 + local_id] + y_off;
        } else {
            y = y_num + 1;
        }

        __syncthreads();

        if( ((global_index_begin_0 + chunk*32 + threadIdx.x) < global_index_end_0) && (y < y_num) ) {
            y_vec[global_index_begin_0 + chunk*32 + threadIdx.x] = y;
        }

    }
}