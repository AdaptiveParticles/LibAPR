#include "ComputeGradientCuda.hpp"
#include "APRParameters.hpp"
#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "data_structures/Mesh/PixelData.hpp"
#include "dsGradient.cuh"

#include "invBspline.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "bsplineXdir.cuh"
#include "bsplineYdir.cuh"
#include "bsplineZdir.cuh"
#include "data_structures/Mesh/downsample.cuh"
#include "algorithm/ComputePullingScheme.cuh"
#include "algorithm/LocalIntensityScaleCuda.h"
#include "algorithm/LocalIntensityScale.cuh"
#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"

// explicit instantiation of handled types
template void getFullPipeline(PixelData<float> &, PixelData<float> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &, int maxLevel);
template void getFullPipeline(PixelData<uint16_t> &, PixelData<uint16_t> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &, int maxLevel);

namespace {
    typedef struct {
        PinnedMemoryUniquePtr<float> bc1;
        PinnedMemoryUniquePtr<float> bc2;
        PinnedMemoryUniquePtr<float> bc3;
        PinnedMemoryUniquePtr<float> bc4;
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
    BsplineParams prepareBsplineStuff(const PixelData<T> &image, float lambda, float tol) {
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

        size_t boundaryLen = sizeof(float) * k0;
        PinnedMemoryUniquePtr<float> bc1{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc2{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc3{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc4{(float*)getPinnedMemory(boundaryLen)};

        //y(0) init
        for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
        //y(1) init
        bc2[1] = impulse_resp_vec_f[0];
        for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k + 1];

        // backward boundaries
        std::vector<float> impulse_resp_vec_b(k0 + 1);
        for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k)
            impulse_resp_vec_b[k] = impulse_resp_back(k, rho, omg, gamma, c0);

        //y(N-1) init
        bc3[0] = impulse_resp_vec_b[1];
        for (size_t k = 0; k < (k0 - 1); ++k) bc3[k + 1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k + 2];
        //y(N) init
        bc4[0] = impulse_resp_vec_b[0];
        for (size_t k = 1; k < k0; ++k) bc4[k] += 2 * impulse_resp_vec_b[k];

        return BsplineParams{
                std::move(bc1),
                std::move(bc2),
                std::move(bc3),
                std::move(bc4),
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

template <typename ImgType, typename T>
void runThreshold(ImgType *cudaImage, T *cudaGrad, size_t x_num, size_t y_num, size_t z_num, float Ip_th, cudaStream_t aStream) {
    dim3 threadsPerBlock(64);
    dim3 numBlocks((x_num * y_num * z_num + threadsPerBlock.x - 1)/threadsPerBlock.x);
    threshold<<<numBlocks,threadsPerBlock, 0, aStream>>>(cudaImage, cudaGrad, x_num * y_num * z_num, Ip_th);
};

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
void runThresholdImg(T *cudaImage, size_t x_num, size_t y_num, size_t z_num, float Ip_th_offset, cudaStream_t aStream) {
    dim3 threadsPerBlock(64);
    dim3 numBlocks((x_num * y_num * z_num + threadsPerBlock.x - 1) / threadsPerBlock.x);
    printCudaDims(threadsPerBlock, numBlocks);
    thresholdImg<<< numBlocks, threadsPerBlock, 0, aStream >>> (cudaImage, x_num * y_num * z_num, Ip_th_offset);
};

template <typename ImgType>
void getGradientCuda(PixelData<ImgType> &image, PixelData<float> &local_scale_temp, PixelData<ImgType> &grad_temp,
                     ImgType *cudaImage, ImgType *cudaGrad, float *cudalocal_scale_temp,
                     float bspline_offset, const APRParameters &par, cudaStream_t aStream, BsplineParams *p) {
    ScopedCudaMemHandler<float> bc1 (p->bc1.get(), p->k0, H2D, aStream);
    ScopedCudaMemHandler<float> bc2 (p->bc2.get(), p->k0, H2D, aStream);
    ScopedCudaMemHandler<float> bc3 (p->bc3.get(), p->k0, H2D, aStream);
    ScopedCudaMemHandler<float> bc4 (p->bc4.get(), p->k0, H2D, aStream);

    runThresholdImg(cudaImage, image.x_num, image.y_num, image.z_num, par.Ip_th + bspline_offset, aStream);

    int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * image.x_num * image.z_num;
    ScopedCudaMemHandler<float> boundary(nullptr, boundaryLen);
 
    runBsplineYdir(cudaImage, image.x_num, image.y_num, image.z_num, bc1.get(), bc2.get(), bc3.get(), bc4.get(), p->k0, p->b1, p->b2, p->norm_factor, boundary.get(), aStream);
    runBsplineXdir(cudaImage, image.x_num, image.y_num, image.z_num, bc1.get(), bc2.get(), bc3.get(), bc4.get(), p->k0, p->b1, p->b2, p->norm_factor, aStream);
    runBsplineZdir(cudaImage, image.x_num, image.y_num, image.z_num, bc1.get(), bc2.get(), bc3.get(), bc4.get(), p->k0, p->b1, p->b2, p->norm_factor, aStream);

    runKernelGradient(cudaImage, cudaGrad, image.x_num, image.y_num, image.z_num, grad_temp.x_num, grad_temp.y_num, par.dx, par.dy, par.dz, aStream);

    runDownsampleMean(cudaImage, cudalocal_scale_temp, image.x_num, image.y_num, image.z_num, aStream);

    runInvBsplineYdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    runInvBsplineXdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    runInvBsplineZdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);

    runThreshold(cudalocal_scale_temp, cudaGrad, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, par.Ip_th, aStream);
}

template <typename ImgType>
void getFullPipeline(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par, int maxLevel) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
{
    ScopedCudaMemHandler<const PixelData<ImgType>> cudaImage(image, H2D, stream);
    ScopedCudaMemHandler<const PixelData<ImgType>> cudaGrad(grad_temp, JUST_ALLOC, stream);
    ScopedCudaMemHandler<PixelData<float>> cudalocal_scale_temp(local_scale_temp, D2H, stream);
    ScopedCudaMemHandler<const PixelData<float>> cudalocal_scale_temp2(local_scale_temp2, JUST_ALLOC, stream);

    float tolerance = 0.0001;
    BsplineParams p = prepareBsplineStuff(image, par.lambda, tolerance);
    getGradientCuda(image, local_scale_temp, grad_temp, cudaImage.get(), cudaGrad.get(), cudalocal_scale_temp.get(), bspline_offset, par, stream, &p);
    runLocalIntensityScalePipeline(local_scale_temp, par, cudalocal_scale_temp.get(), cudalocal_scale_temp2.get(), stream);
    float min_dim = std::min(par.dy, std::min(par.dx, par.dz));
    float level_factor = pow(2, maxLevel) * min_dim;
    const float mult_const = level_factor/par.rel_error;
    runComputeLevels(cudaGrad.get(), cudalocal_scale_temp.get(), grad_temp.mesh.size(), mult_const, stream);
}
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

// =================================================== TEST helpers
// TODO: should be moved somewhere

// explicit instantiation of handled types
template void cudaFilterBsplineFull(PixelData<float> &, float, float, TypeOfRecBsplineFlags);
template <typename ImgType>
void cudaFilterBsplineFull(PixelData<ImgType> &input, float lambda, float tolerance, TypeOfRecBsplineFlags flags) {
    cudaStream_t  aStream = 0;

    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);
    ScopedCudaMemHandler<float> bc1(p.bc1.get(), p.k0, H2D);
    ScopedCudaMemHandler<float> bc2(p.bc2.get(), p.k0, H2D);
    ScopedCudaMemHandler<float> bc3(p.bc3.get(), p.k0, H2D);
    ScopedCudaMemHandler<float> bc4(p.bc4.get(), p.k0, H2D);
    ScopedCudaMemHandler<PixelData<float>> cudaInput(input, D2H + H2D);

    if (flags & BSPLINE_Y_DIR) {
        int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
        ScopedCudaMemHandler<float> boundary(nullptr, boundaryLen); // allocate memory on device
        runBsplineYdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, bc1.get(), bc2.get(), bc3.get(), bc4.get(), p.k0, p.b1, p.b2, p.norm_factor, boundary.get(), aStream);
    }
    if (flags & BSPLINE_X_DIR) {
        runBsplineXdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, bc1.get(), bc2.get(), bc3.get(), bc4.get(), p.k0, p.b1, p.b2, p.norm_factor, aStream);
    }
    if (flags & BSPLINE_Z_DIR) {
        runBsplineZdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, bc1.get(), bc2.get(), bc3.get(), bc4.get(), p.k0, p.b1, p.b2, p.norm_factor, aStream);
    }
}

// explicit instantiation of handled types
template void cudaInverseBspline(PixelData<float> &, TypeOfInvBsplineFlags);
template <typename ImgType>
void cudaInverseBspline(PixelData<ImgType> &input, TypeOfInvBsplineFlags flags) {
    ScopedCudaMemHandler<PixelData<ImgType>> cudaInput(input, H2D + D2H);

    if (flags & INV_BSPLINE_Y_DIR) {
        runInvBsplineYdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, 0);
    }
    if (flags & INV_BSPLINE_X_DIR) {
        runInvBsplineXdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, 0);
    }
    if (flags & INV_BSPLINE_Z_DIR) {
        runInvBsplineZdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, 0);
    }
}

// explicit instantiation of handled types
template void computeLevelsCuda(const PixelData<float> &, PixelData<float> &, int, float, float, float, float);
template <typename ImageType>
void computeLevelsCuda(const PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, int maxLevel, float relError,  float dx, float dy, float dz) {
    ScopedCudaMemHandler<const PixelData<ImageType>> cudaGrad(grad_temp, H2D);
    ScopedCudaMemHandler<PixelData<float>> cudaLis(local_scale_temp, D2H + H2D);

    float min_dim = std::min(dy, std::min(dx, dz));
    float level_factor = pow(2, maxLevel) * min_dim;
    const float mult_const = level_factor/relError;
    cudaStream_t aStream = 0;
    runComputeLevels(cudaGrad.get(), cudaLis.get(), grad_temp.mesh.size(), mult_const, aStream);
}

// explicit instantiation of handled types
template void getGradient(PixelData<float> &, PixelData<float> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &);
template <typename ImgType>
void getGradient(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par) {
    ScopedCudaMemHandler<PixelData<ImgType>> cudaImage(image, D2H + H2D);
    ScopedCudaMemHandler<PixelData<ImgType>> cudaGrad(grad_temp, H2D + D2H);
    ScopedCudaMemHandler<PixelData<float>> cudalocal_scale_temp(local_scale_temp, D2H);
    ScopedCudaMemHandler<PixelData<float>> cudalocal_scale_temp2(local_scale_temp2, D2H);

    float tolerance = 0.0001;
    BsplineParams p = prepareBsplineStuff(image, par.lambda, tolerance);
    getGradientCuda(image, local_scale_temp, grad_temp, cudaImage.get(), cudaGrad.get(), cudalocal_scale_temp.get(), bspline_offset, par, 0, &p);
}

// explicit instantiation of handled types
template void thresholdImg(PixelData<float> &, const float);
template <typename T>
void thresholdImg(PixelData<T> &image, const float threshold) {
    ScopedCudaMemHandler<PixelData<T>> cudaImage(image, D2H + H2D);

    runThresholdImg(cudaImage.get(), image.x_num, image.y_num, image.z_num, threshold, 0);
}

// explicit instantiation of handled types
template void thresholdGradient(PixelData<float> &, const PixelData<float> &, const float);
template <typename T>
void thresholdGradient(PixelData<float> &output, const PixelData<T> &input, const float Ip_th) {
    ScopedCudaMemHandler<const PixelData<T>> cudaInput(input, H2D);
    ScopedCudaMemHandler<PixelData<T>> cudaOutput(output, H2D + D2H);

    runThreshold(cudaInput.get(), cudaOutput.get(), input.x_num, input.y_num, input.z_num, Ip_th, 0);
}

void cudaDownsampledGradient(PixelData<float> &input, PixelData<float> &grad, const float hx, const float hy, const float hz) {
    ScopedCudaMemHandler<PixelData<float>> cudaInput(input, H2D + D2H);
    ScopedCudaMemHandler<PixelData<float>> cudaGrad(grad, D2H);

    runKernelGradient(cudaInput.get(), cudaGrad.get(), input.x_num, input.y_num, input.z_num, grad.x_num, grad.y_num, hx, hy, hz, 0);
}
