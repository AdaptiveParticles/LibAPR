//
// Created by cheesema on 2019-07-08.
//

#ifndef LIBAPR_GPUAPRTEST_CUH_HPP
#define LIBAPR_GPUAPRTEST_CUH_HPP

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
__global__ void testKernel(T *input) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

    input[idx] = idx;

}

bool run_simple_test(){

    std::vector<float> temp;
    temp.resize(100,0);

    ScopedCudaMemHandler<float*, H2D> temp_gpu(temp.data(),100);

    temp_gpu.copyH2D();

    std::cout << temp_gpu.getSize() << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threadsPerBlock(64);
    dim3 numBlocks((temp.size() + threadsPerBlock.x - 1)/threadsPerBlock.x);

    testKernel<<< numBlocks, threadsPerBlock >>>(temp_gpu.get());

    temp_gpu.copyD2H();

    for (int i = 0; i < temp.size(); ++i) {

        if(temp[i]!=i){
            return false;
        }
    }


    return true;

}




#endif //LIBAPR_GPUAPRTEST_CUH_HPP
