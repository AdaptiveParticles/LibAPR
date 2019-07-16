//
// Created by cheesema on 2019-07-08.
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"
#include <chrono>
#include <cstdint>

#include "GPUAPR.hpp"

/**
 * Thresholds output basing on input values. When input is <= thresholdLevel then output is set to 0 and is not changed otherwise.
 * @param input
 * @param output
 * @param length - len of input/output arrays
 * @param thresholdLevel
 */
template <typename T>
__global__ void testKernel(T *input, const uint16_t* y_vec, const uint64_t* xz_end_vec, const uint64_t* level_xz_vec, unsigned int level, uint16_t x_num, uint16_t z_num) {

    /// give in the x_num, y_num, z_num vectors and do it this way?
    size_t x_idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t z_idx = (size_t)blockDim.z * blockIdx.z + threadIdx.z;

    if(x_idx >= x_num) { return; }
    if(z_idx >= z_num) { return; }

    uint64_t offset = x_idx + z_idx * x_num;

    //printf("level = %u\n", level);
    uint64_t level_start = level_xz_vec[level];

    uint64_t xz_start = level_start + offset;

    uint64_t begin_index = xz_end_vec[xz_start-1];
    uint64_t end_index = xz_end_vec[xz_start];

    for(uint64_t i = begin_index; i < end_index; ++i) {
        uint16_t y_idx = y_vec[i];
        input[i] = z_idx + x_idx + y_idx + level;
    }
}

void compute_spatial_info_gpu(GPUAccessHelper& access, std::vector<uint64_t>& temp){

    temp.resize(access.total_number_particles(), 0);

    ScopedCudaMemHandler<uint64_t*, H2D> temp_gpu(temp.data(), access.total_number_particles());

    temp_gpu.copyH2D();

    std::cout << "gpu size: " << temp_gpu.getSize() << " cpu size: " << temp.size() << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    for( unsigned int level = access.level_max(); level >= access.level_min(); --level) {
        dim3 threadsPerBlock(8, 1, 8);
        dim3 numBlocks( (access.x_num(level) + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        1,
                        (access.z_num(level) + threadsPerBlock.z - 1) / threadsPerBlock.z);

        testKernel << < numBlocks, threadsPerBlock >> >(temp_gpu.get(), access.get_y_vec_ptr(), access.get_xz_end_vec_ptr(), access.get_level_xz_vec_ptr(), level, access.x_num(level), access.z_num(level));
    }
    temp_gpu.copyD2H();

}


template <typename T>
__global__ void simpleKernel(T *input, uint64_t size) {

    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= size) { return; }

    input[idx] = idx;
}

void run_simple_test(std::vector<uint64_t>& temp, uint64_t size) {

    temp.resize(size, 0);

    ScopedCudaMemHandler<uint64_t*, H2D> temp_gpu(temp.data(), size);
    temp_gpu.copyH2D();

    std::cout << "gpu size: " << temp_gpu.getSize() << " cpu size: " << temp.size() << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threadsPerBlock(64);
    dim3 numBlocks((temp.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);

    simpleKernel << < numBlocks, threadsPerBlock >> > (temp_gpu.get(), size);

    temp_gpu.copyD2H();
}


template <typename T, typename S>
__global__ void copyKernel(T *input, S* output, uint64_t size) {

    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= size) { return; }

    output[idx] = (S) input[idx];
}

void check_access_vectors(GPUAccessHelper& access, std::vector<uint16_t>& y_vec_out, std::vector<uint64_t>& xz_end_vec_out, std::vector<uint64_t>& level_xz_vec_out) {

    y_vec_out.resize(access.linearAccess->y_vec.size(), 0);
    xz_end_vec_out.resize(access.linearAccess->xz_end_vec.size(), 0);
    level_xz_vec_out.resize(access.linearAccess->level_xz_vec.size(), 0);

    for(uint64_t i = 0; i < level_xz_vec_out.size(); ++i) {
        level_xz_vec_out[i] = i;
    }


    ScopedCudaMemHandler<uint16_t*, H2D> check_array_y(y_vec_out.data(), y_vec_out.size());
    ScopedCudaMemHandler<uint64_t*, H2D> check_array_xz(xz_end_vec_out.data(), xz_end_vec_out.size());
    ScopedCudaMemHandler<uint64_t*, H2D> check_array_lvl(level_xz_vec_out.data(), level_xz_vec_out.size());

    check_array_y.copyH2D();
    check_array_xz.copyH2D();
    check_array_lvl.copyH2D();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threadsPerBlock(64);
    dim3 numBlocks((level_xz_vec_out.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);

    copyKernel<< < numBlocks, threadsPerBlock >> > (access.get_level_xz_vec_ptr(), check_array_lvl.get(), level_xz_vec_out.size());

    numBlocks.x = (xz_end_vec_out.size() + threadsPerBlock.x - 1) / threadsPerBlock.x;
    copyKernel<< < numBlocks, threadsPerBlock >> > (access.get_xz_end_vec_ptr(), check_array_xz.get(), xz_end_vec_out.size());

    numBlocks.x = (y_vec_out.size() + threadsPerBlock.x - 1) / threadsPerBlock.x;
    copyKernel<< < numBlocks, threadsPerBlock >> > (access.get_y_vec_ptr(), check_array_y.get(), y_vec_out.size());

    check_array_y.copyD2H();
    check_array_xz.copyD2H();
    check_array_lvl.copyD2H();
}
