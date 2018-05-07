#include "MeshDataCuda.h"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "misc/CudaTools.hpp"

#include "downsample.cuh"

// explicit instantiation of handled types
template void downsampleMeanCuda(const MeshData<float>&, MeshData<float>&);
template void downsampleMaxCuda(const MeshData<float>&, MeshData<float>&);

template <typename T, typename S>
void downsampleMeanCuda(const MeshData<T> &input, MeshData<S> &output) {
    APRTimer timer(true);

    timer.start_timer("cuda: memory alloc + data transfer to device");

    size_t inputSize = input.mesh.size() * sizeof(T);
    T *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);

    size_t outputSize = output.mesh.size() * sizeof(float);
    float *cudaOutput;
    cudaMalloc(&cudaOutput, outputSize);
    cudaMemcpy(cudaOutput, output.mesh.get(), outputSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    timer.start_timer("cuda: calculations on device");
    dim3 threadsPerBlock(1, 64, 1);
    dim3 numBlocks(((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x + 1) / 2,
                   (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   ((input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z + 1) / 2);
    printCudaDims(threadsPerBlock, numBlocks);

    downsampleMeanKernel<<<numBlocks,threadsPerBlock>>>(cudaInput, cudaOutput, input.x_num, input.y_num, input.z_num);
    waitForCuda();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaFree(cudaInput);
    cudaMemcpy((void*)output.mesh.get(), cudaOutput, outputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaOutput);
    timer.stop_timer();
};

template <typename T, typename S>
void downsampleMaxCuda(const MeshData<T> &input, MeshData<S> &output) {
    APRTimer timer(true);

    timer.start_timer("cuda: memory alloc + data transfer to device");

    size_t inputSize = input.mesh.size() * sizeof(T);
    T *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);

    size_t outputSize = output.mesh.size() * sizeof(float);
    float *cudaOutput;
    cudaMalloc(&cudaOutput, outputSize);
    cudaMemcpy(cudaOutput, output.mesh.get(), outputSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    timer.start_timer("cuda: calculations on device");
    dim3 threadsPerBlock(1, 64, 1);
    dim3 numBlocks(((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x + 1) / 2,
                   (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   ((input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z + 1) / 2);
    printCudaDims(threadsPerBlock, numBlocks);

    downsampleMaxKernel<<<numBlocks,threadsPerBlock>>>(cudaInput, cudaOutput, input.x_num, input.y_num, input.z_num);
    waitForCuda();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaFree(cudaInput);
    cudaMemcpy((void*)output.mesh.get(), cudaOutput, outputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaOutput);
    timer.stop_timer();
};
