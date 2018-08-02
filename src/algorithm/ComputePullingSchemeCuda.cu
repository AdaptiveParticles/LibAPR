//
// Created by Krzysztof Gonciarz on 8/1/18.
//

#include "ComputePullingSchemeCuda.h"
#include "misc/CudaTools.hpp"


// explicit instantiation of handled types
template void computeLevelsCuda(const PixelData<float> &, PixelData<float> &, int, float, float, float, float);

template void gradDivLocalIntensityScale(const float *grad, float *lis, size_t len, float mult_const);
template void gradDivLocalIntensityScale(const uint16_t *grad, float *lis, size_t len, float mult_const);
template void gradDivLocalIntensityScale(const uint8_t *grad, float *lis, size_t len, float mult_const);

template <typename T>
__global__ void gradDivLocalIntensityScaleKernel(const T *grad, float *lis, size_t len, float mult_const) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        //divide gradient magnitude by Local Intensity Scale (first step in calculating the Local Resolution Estimate L(y), minus constants)
        uint32_t d = (grad[idx] / lis[idx]) * mult_const;
        //incorporate other factors and compute the level of the Particle Cell, effectively construct LPC L_n
        lis[idx] = (d == 0) ? 0 : 31 - __clz(d); // fast log2
    }
}

template <typename T>
void gradDivLocalIntensityScale(const T *grad, float *lis, size_t len, float mult_const) {
    CudaTimer timer(true, "levelsCompute");
    timer.start_timer("levels start");
    dim3 threadsPerBlock(64);
    dim3 numBlocks((len + threadsPerBlock.x - 1) / threadsPerBlock.x);
    gradDivLocalIntensityScaleKernel <<< numBlocks, threadsPerBlock >>> (grad, lis, len, mult_const);
    timer.stop_timer();
}

template <typename ImageType>
void computeLevelsCuda(const PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, int maxLevel, float relError,  float dx, float dy, float dz) {
    CudaTimer timer(true, "computeLevelsCuda");
    // Host -> Device transfers
    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t gradSize = grad_temp.mesh.size() * sizeof(ImageType);
    ImageType *cudaGrad = nullptr;
    cudaMalloc(&cudaGrad, gradSize);
    cudaMemcpy(cudaGrad, grad_temp.mesh.get(), gradSize, cudaMemcpyHostToDevice);
    size_t lisSize = local_scale_temp.mesh.size() * sizeof(float);
    float *cudaLis = nullptr;
    cudaMalloc(&cudaLis, lisSize);
    cudaMemcpy(cudaLis, local_scale_temp.mesh.get(), lisSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    // Processing on GPU
    timer.start_timer("cuda: processing on GPU");
    float min_dim = std::min(dy, std::min(dx, dz));
    float level_factor = pow(2, maxLevel) * min_dim;
    const float mult_const = level_factor/relError;
    gradDivLocalIntensityScale(cudaGrad, cudaLis, grad_temp.mesh.size(), mult_const);
    timer.stop_timer();

    // Device -> Host transfers
    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpy((void*)local_scale_temp.mesh.get(), cudaLis, lisSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaLis);
    cudaFree(cudaGrad);
    timer.stop_timer();
}
