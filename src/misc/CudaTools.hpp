//
// Created by Krzysztof Gonciarz on 4/12/18.
//

#ifndef LIBAPR_CUDATOOLS_HPP
#define LIBAPR_CUDATOOLS_HPP


#ifdef APR_USE_CUDA

#include <cuda_runtime.h>
#include <iostream>

inline void waitForCuda() {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
}

inline void printCudaDims(const dim3 &threadsPerBlock, const dim3 &numBlocks) {
    std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
    std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
}

#endif

#endif //LIBAPR_CUDATOOLS_HPP
