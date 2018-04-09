#include "MeshDataCuda.h"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

template <typename T, typename S>
__global__ void downsampleMeanKernel(const T *input, S *output, size_t x_num, size_t y_num, size_t z_num) {
    const size_t xi = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
    const size_t zi = ((blockIdx.z * blockDim.z) + threadIdx.z) * 2;
    if (xi >= x_num || zi >= z_num) return;

    size_t yi = ((blockIdx.y * blockDim.y) + threadIdx.y);
    if (yi == y_num && yi % 2 == 1) {
        // In case when y is odd we need last element to pair with last even y (boundary in y-dir)
        yi = y_num - 1;
    }
    else if (yi >= y_num) {
        return;
    }

    // Handle boundary in x/y direction
    int xs =  xi + 1 > x_num - 1 ? 0 : 1;
    int zs =  zi + 1 > z_num - 1 ? 0 : 1;

    // Index of first element
    size_t idx = (zi * x_num + xi) * y_num + yi;

    // Go through all elements in 2x2
    T v = input[idx];
    v +=  input[idx + xs * y_num];
    v +=  input[idx +              zs * x_num * y_num];
    v +=  input[idx + xs * y_num + zs * x_num * y_num];

    // Get data from odd thread to even one
    const int workerIdx = threadIdx.y;
    T a = __shfl_sync(__activemask(), v, workerIdx + 1, blockDim.y);

    // downsampled dimensions twice smaller (rounded up)

    if (workerIdx % 2 == 0) {
        // Finish calculations in even thread completing whole 2x2x2 cube.
        v += a;

        v /= 8.0; // calculate mean by dividing sum by 8

        // store result in downsampled mesh
        const size_t x_num_ds = ceilf(x_num/2.0);
        const size_t y_num_ds = ceilf(y_num/2.0);
        const size_t dsIdx = (zi/2 * x_num_ds + xi/2) * y_num_ds + yi/2;
        output[dsIdx] = v;
    }
}

template <typename T, typename S>
__global__ void downsampleMaxKernel(const T *input, S *output, size_t x_num, size_t y_num, size_t z_num) {
    const size_t xi = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
    const size_t zi = ((blockIdx.z * blockDim.z) + threadIdx.z) * 2;
    if (xi >= x_num || zi >= z_num) return;

    size_t yi = ((blockIdx.y * blockDim.y) + threadIdx.y);
    if (yi == y_num && yi % 2 == 1) {
        // In case when y is odd we need last element to pair with last even y (boundary in y-dir)
        yi = y_num - 1;
    }
    else if (yi >= y_num) {
        return;
    }

    // Handle boundary in x/y direction
    int xs =  xi + 1 > x_num - 1 ? 0 : 1;
    int zs =  zi + 1 > z_num - 1 ? 0 : 1;

    // Index of first element
    size_t idx = (zi * x_num + xi) * y_num + yi;

    // Go through all elements in 2x2
    T v = input[idx];
    v =  max(input[idx + xs * y_num], v);
    v =  max(input[idx +              zs * x_num * y_num], v);
    v =  max(input[idx + xs * y_num + zs * x_num * y_num], v);

    // Get data from odd thread to even one
    const int workerIdx = threadIdx.y;
    T a = __shfl_sync(__activemask(), v, workerIdx + 1, blockDim.y);

    // downsampled dimensions twice smaller (rounded up)

    if (workerIdx % 2 == 0) {
        // Finish calculations in even thread completing whole 2x2x2 cube.
        v = max(a, v);

        // store result in downsampled mesh
        const size_t x_num_ds = ceilf(x_num/2.0);
        const size_t y_num_ds = ceilf(y_num/2.0);
        const size_t dsIdx = (zi/2 * x_num_ds + xi/2) * y_num_ds + yi/2;
        output[dsIdx] = v;
    }
}

namespace {
    void waitForCuda() {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
    }

    void emptyCallForTemplateInstantiation() {
        MeshData<float> f = MeshData<float>(0, 0, 0);
        MeshData<uint16_t> u16 = MeshData<uint16_t>(0, 0, 0);
        MeshData<uint8_t> u8 = MeshData<uint8_t>(0, 0, 0);

        downsampleMeanCuda(f, f);
        downsampleMeanCuda(f, u16);
        downsampleMeanCuda(f, u8);

        downsampleMaxCuda(f, f);
        downsampleMaxCuda(f, u16);
        downsampleMaxCuda(f, u8);
    }

    void printCudaDims(const dim3 &threadsPerBlock, const dim3 &numBlocks) {
        std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
        std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
    }
}

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
    dim3 threadsPerBlock(1, 32, 1);
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
    dim3 threadsPerBlock(1, 32, 1);
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