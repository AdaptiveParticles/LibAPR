#include "ComputeGradientCuda.hpp"
#include "APRParameters.hpp"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "data_structures/Mesh/PixelData.hpp"
#include "dsGradient.cuh"

#include "algorithm/ComputeInverseCubicBsplineCuda.h"
#include "algorithm/ComputeBsplineRecursiveFilterCuda.h"
#include "invBspline.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "bsplineXdir.cuh"
#include "bsplineYdir.cuh"
#include "bsplineZdir.cuh"
#include "misc/CudaTools.hpp"
#include "data_structures/Mesh/downsample.cuh"
#include "algorithm/LocalIntensityScaleCuda.h"

// explicit instantiation of handled types
template void getGradient(PixelData<float> &, PixelData<float> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &);
template void thresholdImg(PixelData<float> &, const float);
template void getFullPipeline(PixelData<float> &, PixelData<float> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &);
template void getFullPipeline(PixelData<uint16_t> &, PixelData<uint16_t> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &);
template void thresholdGradient(PixelData<float> &, const PixelData<float> &, const float);


void cudaDownsampledGradient(const PixelData<float> &input, PixelData<float> &grad, const float hx, const float hy, const float hz) {
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
    runKernelGradient(cudaInput, cudaGrad, input.x_num, input.y_num, input.z_num, grad.x_num, grad.y_num, hx, hy, hz);
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
    BsplineParams prepareBsplineStuff(PixelData<T> &image, float lambda, float tol) {
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


/**
 * Thresholds output basing on input values. When input is <= thresholdLevel then output is set to 0 and is not changed otherwise.
 * @param input
 * @param output
 * @param length - len of input/output arrays
 * @param thresholdLevel
 */
template <typename T, typename S>
__global__ void threshold(const T *input, S *output, size_t length, float thresholdLevel) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < length) {
        if (input[idx] <= thresholdLevel) { output[idx] = 0; }
    }
}

template <typename T>
void thresholdGradient(PixelData<float> &output, const PixelData<T> &input, const float Ip_th) {
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

/**
 * Thresholds input array to have minimum thresholdLevel.
 * @param input
 * @param length - len of input/output arrays
 * @param thresholdLevel
 */
template <typename T>
__global__ void thresholdImg(T *input, size_t length, float thresholdLevel) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < length) {
        if (input[idx] < thresholdLevel) { input[idx] = thresholdLevel; }
    }
}

template <typename T>
void thresholdImg(PixelData<T> &image, const float threshold) {
    APRTimer timer(true);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t imageSize = image.mesh.size() * sizeof(T);
    T *cudaImage;
    cudaMalloc(&cudaImage, imageSize);
    cudaMemcpy(cudaImage, image.mesh.get(), imageSize, cudaMemcpyHostToDevice);
    timer.stop_timer();

    timer.start_timer("cuda: calculations on device");
    dim3 threadsPerBlock(64);
    dim3 numBlocks((image.x_num * image.y_num * image.z_num + threadsPerBlock.x - 1)/threadsPerBlock.x);
    printCudaDims(threadsPerBlock, numBlocks);
    thresholdImg<<<numBlocks,threadsPerBlock>>>(cudaImage, image.mesh.size(), threshold);
    waitForCuda();
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpy((void*)image.mesh.get(), cudaImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaImage);
    timer.stop_timer();
}

template <typename ImgType>
void getGradientCuda(PixelData<ImgType> &image, PixelData<float> &local_scale_temp, PixelData<ImgType> &grad_temp,
                     ImgType *cudaImage, ImgType *cudaGrad, float *cudalocal_scale_temp,
                     float bspline_offset, const APRParameters &par) {
    CudaTimer timer(true, "getGradientCuda");
    {
        timer.start_timer("threshold");
        dim3 threadsPerBlock(64);
        dim3 numBlocks((image.x_num * image.y_num * image.z_num + threadsPerBlock.x - 1) / threadsPerBlock.x);
        printCudaDims(threadsPerBlock, numBlocks);
        thresholdImg<<< numBlocks, threadsPerBlock >>> (cudaImage, image.mesh.size(), par.Ip_th + bspline_offset);
        timer.stop_timer();
    }
    {
        timer.start_timer("smooth bspline");
        PixelData<ImgType> &input = image;
        float lambda = par.lambda;
        float tolerance = 0.0001;
        BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

        thrust::device_vector<float> d_bc1(p.bc1);
        thrust::device_vector<float> d_bc2(p.bc2);
        thrust::device_vector<float> d_bc3(p.bc3);
        thrust::device_vector<float> d_bc4(p.bc4);
        float *bc1= raw_pointer_cast(d_bc1.data());
        float *bc2= raw_pointer_cast(d_bc2.data());
        float *bc3= raw_pointer_cast(d_bc3.data());
        float *bc4= raw_pointer_cast(d_bc4.data());
        float *boundary;
        uint16_t flags = 0xff;
        if (flags & BSPLINE_Y_DIR) {
            int boundaryLen = sizeof(float) * (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
            cudaMalloc(&boundary, boundaryLen);
        }

        if (flags & BSPLINE_Y_DIR) {
            dim3 threadsPerBlock(numOfThreads);
            dim3 numBlocks((image.x_num * input.z_num + threadsPerBlock.x - 1) / threadsPerBlock.x);
            printCudaDims(threadsPerBlock, numBlocks);
            size_t sharedMemSize = (2 /*bc vectors*/) * (p.k0) * sizeof(float) + numOfThreads * (p.k0) * sizeof(ImgType);
            bsplineYdirBoundary<ImgType> <<< numBlocks, threadsPerBlock, sharedMemSize >>> (cudaImage, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, boundary);
            sharedMemSize = numOfThreads * blockWidth * sizeof(ImgType);
            bsplineYdirProcess<ImgType> <<< numBlocks, threadsPerBlock, sharedMemSize >>> (cudaImage, input.x_num, input.y_num, input.z_num, p.k0, p.b1, p.b2, p.norm_factor, boundary);
            cudaFree(boundary);
        }
        constexpr int numOfWorkersYdir = 64;
        if (flags & BSPLINE_X_DIR) {
            dim3 threadsPerBlockX(1, numOfWorkersYdir, 1);
            dim3 numBlocksX(1,
                            (input.y_num + threadsPerBlockX.y - 1) / threadsPerBlockX.y,
                            (input.z_num + threadsPerBlockX.z - 1) / threadsPerBlockX.z);
            printCudaDims(threadsPerBlockX, numBlocksX);
            bsplineXdir<ImgType> <<< numBlocksX, threadsPerBlockX >>> (cudaImage, input.x_num, input.y_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
        }
        if (flags & BSPLINE_Z_DIR) {
            dim3 threadsPerBlockZ(1, numOfWorkersYdir, 1);
            dim3 numBlocksZ(1,
                            (input.y_num + threadsPerBlockZ.y - 1) / threadsPerBlockZ.y,
                            (input.x_num + threadsPerBlockZ.x - 1) / threadsPerBlockZ.x);
            printCudaDims(threadsPerBlockZ, numBlocksZ);
            bsplineZdir<ImgType> <<< numBlocksZ, threadsPerBlockZ >>> (cudaImage, input.x_num, input.y_num, input.z_num, bc1, bc2, bc3, bc4, p.k0, p.b1, p.b2, p.norm_factor);
        }
        timer.stop_timer();
    }
    {
        timer.start_timer("downsampled_gradient_magnitude");
        PixelData<ImgType> &input = image;
        runKernelGradient(cudaImage, cudaGrad, input.x_num, input.y_num, input.z_num, grad_temp.x_num, grad_temp.y_num, par.dx, par.dy, par.dz);
        timer.stop_timer();
    }
    {
        timer.start_timer("down-sample_b-spline");
        PixelData<ImgType> &input = image;
        dim3 threadsPerBlock(1, 64, 1);
        dim3 numBlocks(((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x + 1) / 2,
                       (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
                       ((input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z + 1) / 2);
        printCudaDims(threadsPerBlock, numBlocks);

        downsampleMeanKernel<<<numBlocks,threadsPerBlock>>>(cudaImage, cudalocal_scale_temp, input.x_num, input.y_num, input.z_num);
        timer.stop_timer();
    }
    {
        TypeOfInvBsplineFlags flags = INV_BSPLINE_ALL_DIR;
        auto &input = local_scale_temp;
        constexpr int numOfWorkers = 32;
        if (flags & INV_BSPLINE_Y_DIR) {
            timer.start_timer("inv y-dir");
            dim3 threadsPerBlock(1, numOfWorkers, 1);
            dim3 numBlocks((input.x_num + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           1,
                           (input.z_num + threadsPerBlock.z - 1) / threadsPerBlock.z);
            printCudaDims(threadsPerBlock, numBlocks);
            invBsplineYdir <<< numBlocks, threadsPerBlock>>> (cudalocal_scale_temp, input.x_num, input.y_num, input.z_num);
            timer.stop_timer();
        }
        if (flags & INV_BSPLINE_X_DIR) {
            timer.start_timer("inv x-dir");
            dim3 threadsPerBlock(1, numOfWorkers, 1);
            dim3 numBlocks(1,
                           (input.y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (input.z_num + threadsPerBlock.z - 1) / threadsPerBlock.z);
            printCudaDims(threadsPerBlock, numBlocks);
            invBsplineXdir <<< numBlocks, threadsPerBlock>>> (cudalocal_scale_temp, input.x_num, input.y_num, input.z_num);
            timer.stop_timer();
        }
        if (flags & INV_BSPLINE_Z_DIR) {
            timer.start_timer("inv z-dir");
            dim3 threadsPerBlock(1, numOfWorkers, 1);
            dim3 numBlocks((input.x_num + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (input.y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           1);
            printCudaDims(threadsPerBlock, numBlocks);
            invBsplineZdir <<< numBlocks, threadsPerBlock>>> (cudalocal_scale_temp, input.x_num, input.y_num, input.z_num);
            timer.stop_timer();
        }
    }
    {
        timer.start_timer("threshold");
        auto &input = local_scale_temp;
        PixelData<ImgType> &output = grad_temp;
        dim3 threadsPerBlock(64);
        dim3 numBlocks((input.x_num * input.y_num * input.z_num + threadsPerBlock.x - 1)/threadsPerBlock.x);
        printCudaDims(threadsPerBlock, numBlocks);

        threshold<<<numBlocks,threadsPerBlock>>>(cudalocal_scale_temp, cudaGrad, output.mesh.size(), par.Ip_th);
        timer.stop_timer();
    }
}

template <typename ImgType>
void getGradient(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par) {
    APRTimer timer(true);

    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t imageSize = image.mesh.size() * sizeof(ImgType);
    ImgType *cudaImage;
    cudaMalloc(&cudaImage, imageSize);
    cudaMemcpy(cudaImage, image.mesh.get(), imageSize, cudaMemcpyHostToDevice);
    size_t gradSize = grad_temp.mesh.size() * sizeof(ImgType);
    ImgType *cudaGrad;
    cudaMalloc(&cudaGrad, gradSize);
    size_t local_scale_tempSize = local_scale_temp.mesh.size() * sizeof(float);
    float *cudalocal_scale_temp;
    cudaMalloc(&cudalocal_scale_temp, local_scale_tempSize);
    float *cudalocal_scale_temp2;
    cudaMalloc(&cudalocal_scale_temp2, local_scale_tempSize);
    timer.stop_timer();


    timer.start_timer("cuda: calculations on device");
    getGradientCuda(image, local_scale_temp, grad_temp, cudaImage, cudaGrad, cudalocal_scale_temp, bspline_offset, par);
    timer.stop_timer();

    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpy((void*)image.mesh.get(), cudaImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaImage);
    cudaMemcpy((void*)grad_temp.mesh.get(), cudaGrad, gradSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaGrad);
    cudaMemcpy((void*)local_scale_temp.mesh.get(), cudalocal_scale_temp, local_scale_tempSize, cudaMemcpyDeviceToHost);
    cudaFree(cudalocal_scale_temp);
    cudaMemcpy((void*)local_scale_temp2.mesh.get(), cudalocal_scale_temp, local_scale_tempSize, cudaMemcpyDeviceToHost);
    cudaFree(cudalocal_scale_temp);
    timer.stop_timer();
}

template <typename ImgType>
void getFullPipeline(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par) {
    CudaTimer timer(true, "getFullPipeline");

    timer.start_timer("Whole processing with transfers");

    // Host -> Device transfers
    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t imageSize = image.mesh.size() * sizeof(ImgType);
    ImgType *cudaImage = nullptr;
    cudaMalloc(&cudaImage, imageSize);
    cudaMemcpy(cudaImage, image.mesh.get(), imageSize, cudaMemcpyHostToDevice);
    size_t gradSize = grad_temp.mesh.size() * sizeof(ImgType);
    ImgType *cudaGrad = nullptr;
    cudaMalloc(&cudaGrad, gradSize);
    size_t local_scale_tempSize = local_scale_temp.mesh.size() * sizeof(float);
    float *cudalocal_scale_temp = nullptr;
    cudaMalloc(&cudalocal_scale_temp, local_scale_tempSize);
    float *cudalocal_scale_temp2 = nullptr;
    cudaMalloc(&cudalocal_scale_temp2, local_scale_tempSize);
    timer.stop_timer();

    // Processing on GPU
    timer.start_timer("cuda: calculations on device PIPELLINE");
    getGradientCuda(image, local_scale_temp, grad_temp, cudaImage, cudaGrad, cudalocal_scale_temp, bspline_offset, par);
    localIntensityScaleCuda(local_scale_temp, par, cudalocal_scale_temp, cudalocal_scale_temp2);
    timer.stop_timer();

    // Device -> Host transfers
    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpy((void*)image.mesh.get(), cudaImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaImage);
    cudaMemcpy((void*)grad_temp.mesh.get(), cudaGrad, gradSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaGrad);
    cudaMemcpy((void*)local_scale_temp.mesh.get(), cudalocal_scale_temp, local_scale_tempSize, cudaMemcpyDeviceToHost);
    cudaFree(cudalocal_scale_temp);
    cudaMemcpy((void*)local_scale_temp2.mesh.get(), cudalocal_scale_temp2, local_scale_tempSize, cudaMemcpyDeviceToHost);
    cudaFree(cudalocal_scale_temp2);
    timer.stop_timer();

    timer.stop_timer();
}
