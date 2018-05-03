#include "ComputeBsplineRecursiveFilterCuda.h"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "bsplineXdir.cuh"
#include "bsplineYdir.cuh"
#include "bsplineZdir.cuh"
#include "misc/CudaTools.hpp"


namespace {
    typedef struct {
        std::vector<float> bc1;
        std::vector<float> bc2;
        std::vector<float> bc3;
        std::vector<float> bc4;
        size_t k0;
        float b1;
        float b2;
        float norm_factor;
    } BsplineParams;

    float impulse_resp(float k, float rho, float omg) {
        //  Impulse Response Function
        return (pow(rho, (std::abs(k))) * sin((std::abs(k) + 1) * omg)) / sin(omg);
    }

    float impulse_resp_back(float k, float rho, float omg, float gamma, float c0) {
        //  Impulse Response Function (nominator eq. 4.8, denominator from eq. 4.7)
        return c0 * pow(rho, std::abs(k)) * (cos(omg * std::abs(k)) + gamma * sin(omg * std::abs(k))) *
               (1.0 / (pow((1 - 2.0 * rho * cos(omg) + pow(rho, 2)), 2)));
    }

    template<typename T>
    BsplineParams prepareBsplineStuff(MeshData<T> &image, float lambda, float tol) {
        // Recursive Filter Implimentation for Smoothing BSplines
        // B-Spline Signal Processing: Part II - Efficient Design and Applications, Unser 1993

        float xi = 1 - 96 * lambda + 24 * lambda * sqrt(3 + 144 * lambda); // eq 4.6
        float rho = (24 * lambda - 1 - sqrt(xi)) / (24 * lambda) *
                    sqrt((1 / xi) * (48 * lambda + 24 * lambda * sqrt(3 + 144 * lambda))); // eq 4.5
        float omg = atan(sqrt((1 / xi) * (144 * lambda - 1))); // eq 4.6

        float c0 = (1 + pow(rho, 2)) / (1 - pow(rho, 2)) * (1 - 2 * rho * cos(omg) + pow(rho, 2)) /
                   (1 + 2 * rho * cos(omg) + pow(rho, 2)); // eq 4.8
        float gamma = (1 - pow(rho, 2)) / (1 + pow(rho, 2)) * (1 / tan(omg)); // eq 4.8

        const float b1 = 2 * rho * cos(omg);
        const float b2 = -pow(rho, 2.0);

        const size_t idealK0Len = ceil(std::abs(log(tol) / log(rho)));
        const size_t minDimension = std::min(image.z_num, std::min(image.x_num, image.y_num));
        const size_t k0 = std::min(idealK0Len, minDimension);

        const float norm_factor = pow((1 - 2.0 * rho * cos(omg) + pow(rho, 2)), 2);
        std::cout << "GPU: xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1
                  << " b2=" << b2 << " k0=" << k0 << " norm_factor=" << norm_factor << std::endl;

        // ------- Calculating boundary conditions

        // forward boundaries
        std::vector<float> impulse_resp_vec_f(k0 + 1);
        for (size_t k = 0; k < impulse_resp_vec_f.size(); ++k) impulse_resp_vec_f[k] = impulse_resp(k, rho, omg);

        //y(0) init
        std::vector<float> bc1(k0, 0);
        for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
        //y(1) init
        std::vector<float> bc2(k0, 0);
        bc2[1] = impulse_resp_vec_f[0];
        for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k + 1];

        // backward boundaries
        std::vector<float> impulse_resp_vec_b(k0 + 1);
        for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k)
            impulse_resp_vec_b[k] = impulse_resp_back(k, rho, omg, gamma, c0);

        //y(N-1) init
        std::vector<float> bc3(k0, 0);
        bc3[0] = impulse_resp_vec_b[1];
        for (size_t k = 0; k < (k0 - 1); ++k) bc3[k + 1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k + 2];
        //y(N) init
        std::vector<float> bc4(k0, 0);
        bc4[0] = impulse_resp_vec_b[0];
        for (size_t k = 1; k < k0; ++k) bc4[k] += 2 * impulse_resp_vec_b[k];


        return BsplineParams{
                bc1,
                bc2,
                bc3,
                bc4,
                k0,
                b1,
                b2,
                norm_factor
        };
    }
}

// explicit instantiation of handled types
template void cudaFilterBsplineFull(MeshData<float> &, float, float, TypeOfRecBsplineFlags);


template <typename ImgType>
void cudaFilterBsplineFull(MeshData<ImgType> &input, float lambda, float tolerance, TypeOfRecBsplineFlags flags) {
    APRTimer timer(true), timerFullPipelilne(true);
    size_t inputSize = input.mesh.size() * sizeof(ImgType);
    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    timer.start_timer("GpuMemTransferHostToDevice");
    thrust::device_vector<float> d_bc1(p.bc1);
    thrust::device_vector<float> d_bc2(p.bc2);
    thrust::device_vector<float> d_bc3(p.bc3);
    thrust::device_vector<float> d_bc4(p.bc4);
    float *bc1= raw_pointer_cast(d_bc1.data());
    float *bc2= raw_pointer_cast(d_bc2.data());
    float *bc3= raw_pointer_cast(d_bc3.data());
    float *bc4= raw_pointer_cast(d_bc4.data());
    ImgType *cudaInput;
    cudaMalloc(&cudaInput, inputSize);
    cudaMemcpy(cudaInput, input.mesh.get(), inputSize, cudaMemcpyHostToDevice);
    float *boundary;
    if (flags & BSPLINE_Y_DIR) {
        int boundaryLen = sizeof(float) * (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
        cudaMalloc(&boundary, boundaryLen);
    }
    timer.stop_timer();

//    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    timerFullPipelilne.start_timer("GpuDeviceTimeFull");
    if (flags & BSPLINE_Y_DIR) {
        timer.start_timer("GpuDeviceTimeYdir");
        dim3 threadsPerBlock(numOfThreads);
        dim3 numBlocks((input.x_num * input.z_num + threadsPerBlock.x - 1) / threadsPerBlock.x);
        printCudaDims(threadsPerBlock, numBlocks);
        size_t sharedMemSize = (2 /*bc vectors*/) * (p.k0) * sizeof(float) + numOfThreads * (p.k0) * sizeof(ImgType);
        bsplineYdirBoundary<ImgType> <<< numBlocks, threadsPerBlock, sharedMemSize >>> (cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, boundary);
        sharedMemSize = numOfThreads * blockWidth * sizeof(ImgType);
        bsplineYdirProcess<ImgType> <<< numBlocks, threadsPerBlock, sharedMemSize >>> (cudaInput, input.x_num, input.y_num, input.z_num, p.k0, p.b1, p.b2, p.norm_factor, boundary);
        waitForCuda();
        cudaFree(boundary);
        timer.stop_timer();
    }
    constexpr int numOfWorkersYdir = 64;
    if (flags & BSPLINE_X_DIR) {
        dim3 threadsPerBlockX(1, numOfWorkersYdir, 1);
        dim3 numBlocksX(1,
                        (input.y_num + threadsPerBlockX.y - 1) / threadsPerBlockX.y,
                        (input.z_num + threadsPerBlockX.z - 1) / threadsPerBlockX.z);
        printCudaDims(threadsPerBlockX, numBlocksX);
        timer.start_timer("GpuDeviceTimeXdir");
        bsplineXdir<ImgType> <<< numBlocksX, threadsPerBlockX >>> (cudaInput, input.x_num, input.y_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
        waitForCuda();
        timer.stop_timer();
    }
    if (flags & BSPLINE_Z_DIR) {
        dim3 threadsPerBlockZ(1, numOfWorkersYdir, 1);
        dim3 numBlocksZ(1,
                        (input.y_num + threadsPerBlockZ.y - 1) / threadsPerBlockZ.y,
                        (input.x_num + threadsPerBlockZ.x - 1) / threadsPerBlockZ.x); // Intentionally x-dim is here (after y) to get good memory coalescing
        printCudaDims(threadsPerBlockZ, numBlocksZ);
        timer.start_timer("GpuDeviceTimeZdir");
        bsplineZdir<ImgType> <<< numBlocksZ, threadsPerBlockZ >>> (cudaInput, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
        waitForCuda();
        timer.stop_timer();
    }
    timerFullPipelilne.stop_timer();

    timer.start_timer("GpuMemTransferDeviceToHost");
    getDataFromKernel(input, inputSize, cudaInput);
    timer.stop_timer();
}
