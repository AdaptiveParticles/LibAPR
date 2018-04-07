#include "ComputeGradientCuda.hpp"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void gradient(float *input, size_t x_num, size_t y_num, size_t z_num, float *grad, size_t x_num_ds, size_t y_num_ds, float hx, float hy, float hz) {
    const int xi = ((blockIdx.x * blockDim.x) + threadIdx.x) * 2;
    const int yi = ((blockIdx.y * blockDim.y) + threadIdx.y) * 2;
    const int zi = ((blockIdx.z * blockDim.z) + threadIdx.z) * 2;
    if (xi >= x_num || yi >= y_num || zi >= z_num) return;

    const size_t xnumynum = x_num * y_num;

    float temp[4][4][4];

    for (int z = 0; z < 4; ++z)
        for (int x = 0; x < 4; ++x)
            for(int y = 0; y < 4; ++y) {
                int xc = xi + x - 1; if (xc < 0) xc = 0; else if (xc >= x_num) xc = x_num - 1;
                int yc = yi + y - 1; if (yc < 0) yc = 0; else if (yc >= y_num) yc = y_num - 1;
                int zc = zi + z - 1; if (zc < 0) zc = 0; else if (zc >= z_num) zc = z_num - 1;
                temp[z][x][y] = *(input + zc * xnumynum + xc * y_num + yc);
            }
    float maxGrad = 0;
    for (int z = 1; z <= 2; ++z)
        for (int x = 1; x <= 2; ++x)
            for(int y = 1; y <= 2; ++y) {
                float xd = (temp[z][x-1][y] - temp[z][x+1][y]) / (2 * hx); xd = xd * xd;
                float yd = (temp[z-1][x][y] - temp[z+1][x][y]) / (2 * hy); yd = yd * yd;
                float zd = (temp[z][x][y-1] - temp[z][x][y+1]) / (2 * hz); zd = zd * zd;
                float gm = __fsqrt_rn(xd + yd + zd);
                if (gm > maxGrad)  maxGrad = gm;
            }

    const size_t idx = zi/2 * x_num_ds * y_num_ds + xi/2 * y_num_ds + yi/2;
    grad[idx] = maxGrad;
}

void cudaDownsampledGradient(const MeshData<float> &input, MeshData<float> &grad, const float hx, const float hy,const float hz) {
    APRTimer timer;
    timer.verbose_flag=true;

    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t inputSize = input.mesh.size() * sizeof(float);
    float *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);

    size_t gradSize = grad.mesh.size() * sizeof(float);
    float *cudaGrad;
    cudaMalloc(&cudaGrad, gradSize);
    timer.stop_timer();

    timer.start_timer("cuda: calculations on device");
    dim3 threadsPerBlock(1, 32, 1);
    dim3 numBlocks((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                   (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   (input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
    std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
    std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;

    gradient<<<numBlocks,threadsPerBlock>>>(cudaInput, input.x_num, input.y_num, input.z_num, cudaGrad, grad.x_num, grad.y_num, hx, hy, hz);
    cudaDeviceSynchronize();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)printf("Error: %s\n", cudaGetErrorString(err));
    cudaMemcpy((void*)input.mesh.get(), cudaInput, inputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaInput);
    cudaMemcpy((void*)grad.mesh.get(), cudaGrad, gradSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaGrad);
    timer.stop_timer();
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

        thresholdGradient(f, f, 0);
        thresholdGradient(f, u16, 0);
        thresholdGradient(f, u8, 0);
    }

    void printCudaDims(const dim3 &threadsPerBlock, const dim3 &numBlocks) {
        std::cout << "Number of blocks  (x/y/z):  " << numBlocks.x << "/" << numBlocks.y << "/" << numBlocks.z << std::endl;
        std::cout << "Number of threads (x/y/z): " << threadsPerBlock.x << "/" << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
    }
}

template <typename T>
__global__ void threshold(const T *input, float *output, size_t length, float thresholdLevel) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < length) {
        if (input[idx] <= thresholdLevel) { output[idx] = 0; }
    }
}

template <typename T>
void thresholdGradient(MeshData<float> &output, const MeshData<T> &input, const float Ip_th) {
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
    dim3 threadsPerBlock(64);
    dim3 numBlocks((input.x_num * input.y_num * input.z_num + threadsPerBlock.x - 1)/threadsPerBlock.x);
    printCudaDims(threadsPerBlock, numBlocks);

    threshold<<<numBlocks,threadsPerBlock>>>(cudaInput, cudaOutput, output.mesh.size(), Ip_th);
    waitForCuda();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaFree(cudaInput);
    cudaMemcpy((void*)output.mesh.get(), cudaOutput, outputSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaOutput);
    timer.stop_timer();
}
