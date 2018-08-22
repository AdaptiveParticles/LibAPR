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
template void getFullPipeline2(PixelData<float> &, PixelData<float> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &, int maxLevel);
template void getFullPipeline2(PixelData<uint16_t> &, PixelData<uint16_t> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &, int maxLevel);

namespace {
    typedef struct {
//        std::vector<float> bc1;
//        std::vector<float> bc2;
//        std::vector<float> bc3;
//        std::vector<float> bc4;
        float *bc1;
        float *bc2;
        float *bc3;
        float *bc4;
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

        float  *bc1 = (float*)getPinnedMemory(sizeof(float) * k0);
        float  *bc2 = (float*)getPinnedMemory(sizeof(float) * k0);
        float  *bc3 = (float*)getPinnedMemory(sizeof(float) * k0);
        float  *bc4 = (float*)getPinnedMemory(sizeof(float) * k0);

        //y(0) init
//        std::vector<float> bc1(k0, 0);
        for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
        //y(1) init
//        std::vector<float> bc2(k0, 0);
        bc2[1] = impulse_resp_vec_f[0];
        for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k + 1];

        // backward boundaries
        std::vector<float> impulse_resp_vec_b(k0 + 1);
        for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k)
            impulse_resp_vec_b[k] = impulse_resp_back(k, rho, omg, gamma, c0);

        //y(N-1) init
//        std::vector<float> bc3(k0, 0);
        bc3[0] = impulse_resp_vec_b[1];
        for (size_t k = 0; k < (k0 - 1); ++k) bc3[k + 1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k + 2];
        //y(N) init
//        std::vector<float> bc4(k0, 0);
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
struct XYZ {
    ImgType *image;
    ImgType *grad;
    float *lis1;
    float *lis2;
    float *bc1, *bc2, *bc3, *bc4;
    BsplineParams p;
    float *boundary;
};

template <typename ImgType>
void generateBsplineParams(XYZ<ImgType> &asdf, const PixelData<ImgType> &input, float lambda, cudaStream_t stream1) {
    float tolerance = 0.0001;
    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    float *bc1, *bc2, *bc3, *bc4;
    cudaMalloc(&bc1, p.k0 * sizeof(float));
    cudaMalloc(&bc2, p.k0 * sizeof(float));
    cudaMalloc(&bc3, p.k0 * sizeof(float));
    cudaMalloc(&bc4, p.k0 * sizeof(float));

    cudaMemcpyAsync(bc1, p.bc1, p.k0 * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(bc2, p.bc2, p.k0 * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(bc3, p.bc3, p.k0 * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(bc4, p.bc4, p.k0 * sizeof(float), cudaMemcpyHostToDevice, stream1);
    asdf.bc1 = bc1;
    asdf.bc2 = bc2;
    asdf.bc3 = bc3;
    asdf.bc4 = bc4;
    asdf.p = p;

    float *boundary;
    int boundaryLen = sizeof(float) * (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
    cudaMalloc(&boundary, boundaryLen);
    asdf.boundary = boundary;


    // TODO: free allocated memory in this funciton!!
}

template <typename ImgType>
void getGradientCuda(PixelData<ImgType> &image, PixelData<float> &local_scale_temp, PixelData<ImgType> &grad_temp,
                     ImgType *cudaImage, ImgType *cudaGrad, float *cudalocal_scale_temp,
                     float bspline_offset, const APRParameters &par, cudaStream_t aStream, XYZ<ImgType> *p) {
    runThresholdImg(cudaImage, image.x_num, image.y_num, image.z_num, par.Ip_th + bspline_offset, aStream);

    runBsplineYdir(cudaImage, image.x_num, image.y_num, image.z_num, p->bc1, p->bc2, p->bc3, p->bc4, p->p.k0, p->p.b1, p->p.b2, p->p.norm_factor, p->boundary, aStream);
    runBsplineXdir(cudaImage, image.x_num, image.y_num, image.z_num, p->bc1, p->bc2, p->bc3, p->bc4, p->p.k0, p->p.b1, p->p.b2, p->p.norm_factor, aStream);
    runBsplineZdir(cudaImage, image.x_num, image.y_num, image.z_num, p->bc1, p->bc2, p->bc3, p->bc4, p->p.k0, p->p.b1, p->p.b2, p->p.norm_factor, aStream);

    runKernelGradient(cudaImage, cudaGrad, image.x_num, image.y_num, image.z_num, grad_temp.x_num, grad_temp.y_num, par.dx, par.dy, par.dz, aStream);

    runDownsampleMean(cudaImage, cudalocal_scale_temp, image.x_num, image.y_num, image.z_num, aStream);

    runInvBsplineYdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    runInvBsplineXdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    runInvBsplineZdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);

    runThreshold(cudalocal_scale_temp, cudaGrad, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, par.Ip_th, aStream);
}

template <typename ImgType>
void getFullPipeline(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par, int maxLevel) {
    CudaTimer timer(true, "cuda: getFullPipeline");

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    timer.start_timer("cuda: Whole processing with transfers");

    // Host -> Device transfers
    timer.start_timer("cuda: memory alloc + data transfer to device");
    size_t imageSize = image.mesh.size() * sizeof(ImgType);
    ImgType *cudaImage = nullptr;
    cudaMalloc(&cudaImage, imageSize);
    cudaMemcpyAsync(cudaImage, image.mesh.get(), imageSize, cudaMemcpyHostToDevice, stream1);
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
    cudaStream_t processingStream = stream1;
    XYZ<ImgType> memoryPack = XYZ<ImgType>{cudaImage, cudaGrad, cudalocal_scale_temp, cudalocal_scale_temp2};
    generateBsplineParams(memoryPack, image, par.lambda, processingStream);
    getGradientCuda(image, local_scale_temp, grad_temp, cudaImage, cudaGrad, cudalocal_scale_temp, bspline_offset, par, processingStream, &memoryPack);
    runLocalIntensityScalePipeline(local_scale_temp, par, cudalocal_scale_temp, cudalocal_scale_temp2, processingStream);
    float min_dim = std::min(par.dy, std::min(par.dx, par.dz));
    float level_factor = pow(2, maxLevel) * min_dim;
    const float mult_const = level_factor/par.rel_error;
    runComputeLevels(cudaGrad, cudalocal_scale_temp, grad_temp.mesh.size(), mult_const, processingStream);
    timer.stop_timer();

    // Device -> Host transfers
    timer.start_timer("cuda: transfer data from device and freeing memory");
    cudaMemcpyAsync((void*)image.mesh.get(), cudaImage, imageSize, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync((void*)grad_temp.mesh.get(), cudaGrad, gradSize, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync((void*)local_scale_temp2.mesh.get(), cudalocal_scale_temp2, local_scale_tempSize, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync((void*)local_scale_temp.mesh.get(), cudalocal_scale_temp, local_scale_tempSize, cudaMemcpyDeviceToHost, stream1);
    cudaFree(cudalocal_scale_temp2);
    cudaFree(cudaImage);
    cudaFree(cudaGrad);
    cudaFree(cudalocal_scale_temp);
    timer.stop_timer();

    timer.stop_timer();

    cudaStreamSynchronize(stream1);
    cudaStreamDestroy(stream1);
}



template <typename ImgType>
void getFullPipeline2(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par, int maxLevel) {
    CudaTimer timer(true, "cuda: getFullPipeline");

    timer.start_timer("copy to vector");
    const int num = 3;

    std::vector<PixelData<ImgType>> inData;
    std::vector<XYZ<ImgType>> inMemory;
    for (int i = 0;i < num; ++i) {
        inData.emplace_back(PixelData<ImgType>(image, true, true));
    }

    timer.stop_timer();

    cudaStream_t streams[num];
    for (int i = 0; i < num; ++i) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    timer.start_timer("cuda: Whole processing with transfers");


    for (int i = 0; i < num; ++i) {
        // Host -> Device transfers
        timer.start_timer("cuda: memory alloc + data transfer to device");
        size_t imageSize = image.mesh.size() * sizeof(ImgType);
        ImgType *cudaImage = nullptr;
        cudaMalloc(&cudaImage, imageSize);
        size_t gradSize = grad_temp.mesh.size() * sizeof(ImgType);
        ImgType *cudaGrad = nullptr;
        cudaMalloc(&cudaGrad, gradSize);
        size_t local_scale_tempSize = local_scale_temp.mesh.size() * sizeof(float);
        float *cudalocal_scale_temp = nullptr;
        cudaMalloc(&cudalocal_scale_temp, local_scale_tempSize);
        float *cudalocal_scale_temp2 = nullptr;
        cudaMalloc(&cudalocal_scale_temp2, local_scale_tempSize);
        timer.stop_timer();

        inMemory.emplace_back( XYZ<ImgType>{cudaImage, cudaGrad, cudalocal_scale_temp, cudalocal_scale_temp2} );
    }
    for (int i = 0; i < num; ++i) {
        cudaStream_t stream1 = streams[i];
        // Host -> Device transfers
        timer.start_timer("cuda: memory alloc + data transfer to device");
        size_t imageSize = image.mesh.size() * sizeof(ImgType);
        ImgType *cudaImage = inMemory[i].image;
        cudaMemcpyAsync(cudaImage, image.mesh.get(), imageSize, cudaMemcpyHostToDevice, stream1);

        generateBsplineParams(inMemory[i], image, par.lambda, stream1);
        timer.stop_timer();
    }

    for (int i = 0; i < num; ++i) {
        cudaStream_t stream1 = streams[i];

        ImgType *cudaImage = inMemory[i].image;
        ImgType *cudaGrad = inMemory[i].grad;
        float *cudalocal_scale_temp = inMemory[i].lis1;
        float *cudalocal_scale_temp2 = inMemory[i].lis2;


        // Processing on GPU
        timer.start_timer("cuda: calculations on device PIPELLINE");
        cudaStream_t processingStream = stream1;
        getGradientCuda(image, local_scale_temp, grad_temp, cudaImage, cudaGrad, cudalocal_scale_temp, bspline_offset,
                        par, processingStream, &inMemory[i]);
        runLocalIntensityScalePipeline(local_scale_temp, par, cudalocal_scale_temp, cudalocal_scale_temp2,
                                       processingStream);
        float min_dim = std::min(par.dy, std::min(par.dx, par.dz));
        float level_factor = pow(2, maxLevel) * min_dim;
        const float mult_const = level_factor / par.rel_error;
        runComputeLevels(cudaGrad, cudalocal_scale_temp, grad_temp.mesh.size(), mult_const, processingStream);
        timer.stop_timer();
    }

    for (int i = 0; i < num; ++i) {
        // Device -> Host transfers
        timer.start_timer("cuda: transfer data from device and freeing memory");
//    cudaMemcpy((void*)image.mesh.get(), cudaImage, imageSize, cudaMemcpyDeviceToHost);
//    cudaMemcpy((void*)grad_temp.mesh.get(), cudaGrad, gradSize, cudaMemcpyDeviceToHost);
//    cudaMemcpy((void*)local_scale_temp2.mesh.get(), cudalocal_scale_temp2, local_scale_tempSize, cudaMemcpyDeviceToHost);
        cudaStream_t stream1 = streams[i];


        ImgType *cudaImage = inMemory[i].image;
        ImgType *cudaGrad = inMemory[i].grad;
        float *cudalocal_scale_temp = inMemory[i].lis1;
        float *cudalocal_scale_temp2 = inMemory[i].lis2;

        size_t local_scale_tempSize = local_scale_temp.mesh.size() * sizeof(float);
        cudaMemcpyAsync((void *) local_scale_temp.mesh.get(), cudalocal_scale_temp, local_scale_tempSize,
                        cudaMemcpyDeviceToHost, stream1);
        cudaFree(cudalocal_scale_temp2);
        cudaFree(cudaImage);
        cudaFree(cudaGrad);
        cudaFree(cudalocal_scale_temp);

        cudaFree(inMemory[i].bc1);
        cudaFree(inMemory[i].bc2);
        cudaFree(inMemory[i].bc3);
        cudaFree(inMemory[i].bc4);

        timer.stop_timer();
    }

    timer.stop_timer();
    for (int i = 0; i < num; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}


// =================================================== TEST helpers
// TODO: should be moved somewhere

// explicit instantiation of handled types
template void cudaFilterBsplineFull(PixelData<float> &, float, float, TypeOfRecBsplineFlags, int);
template <typename ImgType>
void cudaFilterBsplineFull(PixelData<ImgType> &input, float lambda, float tolerance, TypeOfRecBsplineFlags flags, int k0Len) {
    cudaStream_t  aStream = 0;

    BsplineParams p = prepareBsplineStuff(input, lambda, tolerance);

    ScopedMemHandler<float> bc1(p.bc1, p.k0, H2D);
    ScopedMemHandler<float> bc2(p.bc2, p.k0, H2D);
    ScopedMemHandler<float> bc3(p.bc3, p.k0, H2D);
    ScopedMemHandler<float> bc4(p.bc4, p.k0, H2D);
    ScopedMemHandler<PixelData<float>> cudaInput(input, D2H + H2D);

    if (flags & BSPLINE_Y_DIR) {
        int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
        ScopedMemHandler<float> boundary(nullptr, boundaryLen);
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
    ScopedMemHandler<PixelData<ImgType>> cudaInput(input, H2D + D2H);

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
    ScopedMemHandler<const PixelData<ImageType>> cudaGrad(grad_temp, H2D);
    ScopedMemHandler<PixelData<float>> cudaLis(local_scale_temp, D2H + H2D);

    // Processing on GPU
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
    ScopedMemHandler<PixelData<ImgType>> cudaImage(image, D2H + H2D);
    ScopedMemHandler<PixelData<ImgType>> cudaGrad(grad_temp, D2H);
    ScopedMemHandler<PixelData<float>> cudalocal_scale_temp(local_scale_temp, D2H);
    ScopedMemHandler<PixelData<float>> cudalocal_scale_temp2(local_scale_temp2, D2H);

    XYZ<ImgType> memoryPack = XYZ<ImgType>{cudaImage.get(), cudaGrad.get(), cudalocal_scale_temp.get(), cudalocal_scale_temp2.get()};
    generateBsplineParams(memoryPack, image, par.lambda, 0);
    getGradientCuda(image, local_scale_temp, grad_temp, cudaImage.get(), cudaGrad.get(), cudalocal_scale_temp.get(), bspline_offset, par, 0, &memoryPack);
}

// explicit instantiation of handled types
template void thresholdImg(PixelData<float> &, const float);
template <typename T>
void thresholdImg(PixelData<T> &image, const float threshold) {
    ScopedMemHandler<PixelData<T>> cudaImage(image, D2H + H2D);

    runThresholdImg(cudaImage.get(), image.x_num, image.y_num, image.z_num, threshold, 0);
}

// explicit instantiation of handled types
template void thresholdGradient(PixelData<float> &, const PixelData<float> &, const float);
template <typename T>
void thresholdGradient(PixelData<float> &output, const PixelData<T> &input, const float Ip_th) {
    ScopedMemHandler<const PixelData<T>> cudaInput(input, H2D);
    ScopedMemHandler<PixelData<T>> cudaOutput(output, H2D + D2H);

    runThreshold(cudaInput.get(), cudaOutput.get(), input.x_num, input.y_num, input.z_num, Ip_th, 0);
}

void cudaDownsampledGradient(PixelData<float> &input, PixelData<float> &grad, const float hx, const float hy, const float hz) {
    ScopedMemHandler<PixelData<float>> cudaInput(input, H2D + D2H);
    ScopedMemHandler<PixelData<float>> cudaGrad(grad, D2H);

    runKernelGradient(cudaInput.get(), cudaGrad.get(), input.x_num, input.y_num, input.z_num, grad.x_num, grad.y_num, hx, hy, hz, 0);
}
