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
#include <chrono>
#include <cstdint>

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

    BsplineParams prepareBsplineStuff(size_t dimensionLen, float lambda, float tol, int maxFilterLen = -1) {
        // Recursive Filter Implimentation for Smoothing BSplines
        // B-Spline Signal Processing: Part II - Efficient Design and Applications, Unser 1993

        float xi = 1 - 96 * lambda + 24 * lambda * sqrt(3 + 144 * lambda); // eq 4.6
        float rho = (24 * lambda - 1 - sqrt(xi)) / (24 * lambda) * sqrt((1 / xi) * (48 * lambda + 24 * lambda * sqrt(3 + 144 * lambda))); // eq 4.5
        float omg = atan(sqrt((1 / xi) * (144 * lambda - 1))); // eq 4.6
        float c0 = (1 + pow(rho, 2)) / (1 - pow(rho, 2)) * (1 - 2 * rho * cos(omg) + pow(rho, 2)) / (1 + 2 * rho * cos(omg) + pow(rho, 2)); // eq 4.8
        float gamma = (1 - pow(rho, 2)) / (1 + pow(rho, 2)) * (1 / tan(omg)); // eq 4.8

        const float b1 = 2 * rho * cos(omg);
        const float b2 = -pow(rho, 2.0);

        const size_t idealK0Len = ceil(std::abs(log(tol)/log(rho)));
        const size_t k0 = maxFilterLen > 0 ? maxFilterLen : idealK0Len;
        const size_t minLen = std::min(idealK0Len, dimensionLen);

        const float norm_factor = pow((1 - 2.0 * rho * cos(omg) + pow(rho, 2)), 2);
        std::cout << "GPU: xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1 << " b2=" << b2 << " k0=" << k0 << " norm_factor=" << norm_factor << std::endl;

        // ------- Calculating boundary conditions

        size_t boundaryLen = sizeof(float) * k0;
        PinnedMemoryUniquePtr<float> bc1{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc2{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc3{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc4{(float*)getPinnedMemory(boundaryLen)};

        // ------- forward boundaries
        std::vector<float> impulse_resp_vec_f(k0 + 1);
        for (size_t k = 0; k < impulse_resp_vec_f.size(); ++k) impulse_resp_vec_f[k] = impulse_resp(k, rho, omg);

        //y(0) init
        for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
        //assumes a constant value at the end of the filter when the required ghost is bigger then the image
        for(size_t k = minLen; k < k0; ++k) bc1[minLen-1] += bc1[k];

        //y(1) init
        for (size_t k = 0; k < k0; ++k) bc2[k] = 0;
        bc2[1] = impulse_resp_vec_f[0];
        for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k + 1];
        for(size_t k = minLen; k < k0; ++k) bc2[minLen-1] += bc2[k];

        // ------- backward boundaries
        std::vector<float> impulse_resp_vec_b(k0 + 1);
        for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k) impulse_resp_vec_b[k] = impulse_resp_back(k, rho, omg, gamma, c0);

        //y(N-1) init
        for (size_t k = 0; k < k0; ++k) bc3[k] = 0;
        bc3[0] = impulse_resp_vec_b[1];
        for (size_t k = 0; k < (k0 - 1); ++k) bc3[k + 1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k + 2];
        for(size_t k = minLen; k < k0; ++k) bc3[minLen-1] += bc3[k];
        //y(N) init
        for (size_t k = 0; k < k0; ++k) bc4[k] = 0;
        bc4[0] = impulse_resp_vec_b[0];
        for (size_t k = 1; k < k0; ++k) bc4[k] += 2 * impulse_resp_vec_b[k];
        for(size_t k = minLen; k < k0; ++k) bc4[minLen-1] += bc4[k];

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
    thresholdImg<<< numBlocks, threadsPerBlock, 0, aStream >>> (cudaImage, x_num * y_num * z_num, Ip_th_offset);
};

template <typename ImgType>
void getGradientCuda(const PixelData<ImgType> &image, PixelData<float> &local_scale_temp,
                     ImgType *cudaImage, ImgType *cudaGrad, float *cudalocal_scale_temp,
                     BsplineParams &px, BsplineParams &py, BsplineParams &pz, float *bc1x, float *bc2x, float *bc3x, float *bc4x, float *bc1y, float *bc2y, float *bc3y, float *bc4y, float *bc1z, float *bc2z, float *bc3z, float *bc4z, float *boundary,
                     float bspline_offset, const APRParameters &par, cudaStream_t aStream) {
// TODO: thresholdin is removed from gradient pipeline in APRConverter - to be investigated where it should be done
//    runThresholdImg(cudaImage, image.x_num, image.y_num, image.z_num, par.Ip_th + bspline_offset, aStream);

    runBsplineYdir(cudaImage, image.x_num, image.y_num, image.z_num, bc1y, bc2y, bc3y, bc4y, py.k0, py.b1, py.b2, py.norm_factor, boundary, aStream);
    runBsplineXdir(cudaImage, image.x_num, image.y_num, image.z_num, bc1x, bc2x, bc3x, bc4x, px.k0, px.b1, px.b2, px.norm_factor, aStream);
    runBsplineZdir(cudaImage, image.x_num, image.y_num, image.z_num, bc1z, bc2z, bc3z, bc4z, pz.k0, pz.b1, pz.b2, pz.norm_factor, aStream);

    runKernelGradient(cudaImage, cudaGrad, image.x_num, image.y_num, image.z_num, local_scale_temp.x_num, local_scale_temp.y_num, par.dx, par.dy, par.dz, aStream);

    runDownsampleMean(cudaImage, cudalocal_scale_temp, image.x_num, image.y_num, image.z_num, aStream);

    runInvBsplineYdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    runInvBsplineXdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    runInvBsplineZdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);

//    runThreshold(cudalocal_scale_temp, cudaGrad, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, par.Ip_th, aStream);

}

class CurrentTime {
    std::chrono::high_resolution_clock m_clock;

public:
    uint64_t milliseconds() {
        return std::chrono::duration_cast<std::chrono::milliseconds>
                (m_clock.now().time_since_epoch()).count();
    }
    uint64_t microseconds() {
        return std::chrono::duration_cast<std::chrono::microseconds>
                (m_clock.now().time_since_epoch()).count();
    }
    uint64_t nanoseconds() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>
                (m_clock.now().time_since_epoch()).count();
    }
};

template <typename U>
template <typename ImgType>
class GpuProcessingTask<U>::GpuProcessingTaskImpl {

    // input data
    const PixelData<ImgType> &iCpuImage;
    PixelData<float> &iCpuLevels;
    const APRParameters &iParameters;
    float iBsplineOffset;
    int iMaxLevel;

    // cuda stuff - memory and stream to be used
    const cudaStream_t iStream;
    ScopedCudaMemHandler<const PixelData<ImgType>, JUST_ALLOC> image;
    ScopedCudaMemHandler<PixelData<ImgType>, JUST_ALLOC> gradient;
    ScopedCudaMemHandler<PixelData<float>, JUST_ALLOC> local_scale_temp;
    ScopedCudaMemHandler<PixelData<float>, JUST_ALLOC> local_scale_temp2;

    // bspline stuff
    const float tolerance = 0.0001;
    BsplineParams params;
    ScopedCudaMemHandler<float*, H2D> bc1;
    ScopedCudaMemHandler<float*, H2D> bc2;
    ScopedCudaMemHandler<float*, H2D> bc3;
    ScopedCudaMemHandler<float*, H2D> bc4;
    const size_t boundaryLen;
    ScopedCudaMemHandler<float*, JUST_ALLOC> boundary;

    /**
     * @return newly created stream
     */
    cudaStream_t getStream() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return stream;
    }

public:

    GpuProcessingTaskImpl(const PixelData<ImgType> &image, PixelData<float> &levels, const APRParameters &parameters, float bspline_offset, int maxLevel) :
        iCpuImage(image),
        iCpuLevels(levels),
        iStream(getStream()),
        image (image, iStream),
        gradient (levels, iStream),
        local_scale_temp (levels, iStream),
        local_scale_temp2 (levels, iStream),
        iParameters(parameters),
        iBsplineOffset(bspline_offset),
        iMaxLevel(maxLevel),
            //TODO: This is just temporary fix to have this compiled but x,y,z must be handled seperately
        params(prepareBsplineStuff(std::min(image.y_num, std::min(image.x_num, image.z_num)), parameters.lambda, tolerance)),
        bc1(params.bc1.get(), params.k0, iStream),
        bc2(params.bc2.get(), params.k0, iStream),
        bc3(params.bc3.get(), params.k0, iStream),
        bc4(params.bc4.get(), params.k0, iStream),
        boundaryLen{(2 /*two first elements*/ + 2 /* two last elements */) * static_cast<size_t>(image.x_num) * static_cast<size_t>(image.z_num)},
        boundary{nullptr, boundaryLen, iStream}
    {
//        std::cout << "\n=============== GpuProcessingTaskImpl ===================\n\n";
        std::cout << iCpuImage << std::endl;
        std::cout << iCpuLevels << std::endl;
        std::cout << "\n\n\n";

    }

    void sendDataToGpu() {
        CurrentTime ct;
        uint64_t start = ct.microseconds();
        image.copyH2D();
        std::cout << "SEND time: " << ct.microseconds() - start << std::endl;
    }

    void getDataFromGpu() {
        CurrentTime ct;
        uint64_t start = ct.microseconds();
        local_scale_temp.copyD2H();
        cudaStreamSynchronize(iStream);
        std::cout << "RCV time: " << ct.microseconds() - start << std::endl;
    }

    void processOnGpu() {
        CurrentTime ct;
        uint64_t start = ct.microseconds();
        // TODO: boundary&params conditions should be passed per dimention (now same used in xyz just to be able to compile).
        getGradientCuda(iCpuImage, iCpuLevels, image.get(), gradient.get(), local_scale_temp.get(),
                        params, params, params, bc1.get(), bc2.get(), bc3.get(), bc4.get(), bc1.get(), bc2.get(), bc3.get(), bc4.get(), bc1.get(), bc2.get(), bc3.get(), bc4.get(), boundary.get(),
                        iBsplineOffset, iParameters, iStream);
        std::cout << "1: " << ct.microseconds() - start << std::endl;
        runLocalIntensityScalePipeline(iCpuLevels, iParameters, local_scale_temp.get(), local_scale_temp2.get(), iStream);
        std::cout << "2: " << ct.microseconds() - start << std::endl;
        float min_dim = std::min(iParameters.dy, std::min(iParameters.dx, iParameters.dz));
        float level_factor = pow(2, iMaxLevel) * min_dim;
        const float mult_const = level_factor/iParameters.rel_error;
        runComputeLevels(gradient.get(), local_scale_temp.get(), iCpuLevels.mesh.size(), mult_const, iStream);
        std::cout << "3: " << ct.microseconds() - start << std::endl;
    }

    ~GpuProcessingTaskImpl() {
        cudaStreamDestroy(iStream);
//        std::cout << "\n============== ~GpuProcessingTaskImpl ===================\n\n";
    }
};

template <typename ImgType>
GpuProcessingTask<ImgType>::GpuProcessingTask(PixelData<ImgType> &image, PixelData<float> &levels, const APRParameters &parameters, float bspline_offset, int maxLevel)
: impl{new GpuProcessingTaskImpl<ImgType>(image, levels, parameters, bspline_offset, maxLevel)} {std::cout << "GpuProcessingTask\n";}

template <typename ImgType>
GpuProcessingTask<ImgType>::~GpuProcessingTask() {std::cout << "~GpuProcessingTask\n";}

template <typename ImgType>
GpuProcessingTask<ImgType>::GpuProcessingTask(GpuProcessingTask&&) = default;

template <typename ImgType>
void GpuProcessingTask<ImgType>::sendDataToGpu() {impl->sendDataToGpu();}

template <typename ImgType>
void GpuProcessingTask<ImgType>::getDataFromGpu() {impl->getDataFromGpu();}

template <typename ImgType>
void GpuProcessingTask<ImgType>::processOnGpu() {impl->processOnGpu();}

template <typename ImgType>
void GpuProcessingTask<ImgType>::doAll() {
    sendDataToGpu();
    processOnGpu();
    getDataFromGpu();
}

// explicit instantiation of handled types
template class GpuProcessingTask<uint16_t>;
template class GpuProcessingTask<float>;

// ================================== TEST helpers ==============
// TODO: should be moved somewhere

// explicit instantiation of handled types
template void cudaFilterBsplineFull(PixelData<float> &, float, float, TypeOfRecBsplineFlags, int);
template <typename ImgType>
void cudaFilterBsplineFull(PixelData<ImgType> &input, float lambda, float tolerance, TypeOfRecBsplineFlags flags, int maxFilterLen) {
    cudaStream_t  aStream = 0;

    BsplineParams px = prepareBsplineStuff(input.x_num, lambda, tolerance, maxFilterLen);
    ScopedCudaMemHandler<float*, H2D> bc1x(px.bc1.get(), px.k0);
    ScopedCudaMemHandler<float*, H2D> bc2x(px.bc2.get(), px.k0);
    ScopedCudaMemHandler<float*, H2D> bc3x(px.bc3.get(), px.k0);
    ScopedCudaMemHandler<float*, H2D> bc4x(px.bc4.get(), px.k0);
    BsplineParams py = prepareBsplineStuff(input.y_num, lambda, tolerance, maxFilterLen);
    ScopedCudaMemHandler<float*, H2D> bc1y(py.bc1.get(), py.k0);
    ScopedCudaMemHandler<float*, H2D> bc2y(py.bc2.get(), py.k0);
    ScopedCudaMemHandler<float*, H2D> bc3y(py.bc3.get(), py.k0);
    ScopedCudaMemHandler<float*, H2D> bc4y(py.bc4.get(), py.k0);
    BsplineParams pz = prepareBsplineStuff(input.z_num, lambda, tolerance, maxFilterLen);
    ScopedCudaMemHandler<float*, H2D> bc1z(pz.bc1.get(), pz.k0);
    ScopedCudaMemHandler<float*, H2D> bc2z(pz.bc2.get(), pz.k0);
    ScopedCudaMemHandler<float*, H2D> bc3z(pz.bc3.get(), pz.k0);
    ScopedCudaMemHandler<float*, H2D> bc4z(pz.bc4.get(), pz.k0);

    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaInput(input);

    auto printVec = [&](std::string str, float *v, int k0){
        std::cout << str;
        for (int i = 0; i < k0; ++i) std::cout << v[i] << " ";
        std::cout << std::endl;
    };

    printVec("BC1: ", px.bc1.get(), px.k0);
    printVec("BC2: ", px.bc2.get(), px.k0);
    printVec("BC3: ", px.bc3.get(), px.k0);
    printVec("BC4: ", px.bc4.get(), px.k0);

    APRTimer timer(true);
    timer.start_timer("GpuDeviceTimeFull");
    if (flags & BSPLINE_Y_DIR) {
        int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
        ScopedCudaMemHandler<float*, JUST_ALLOC> boundary(nullptr, boundaryLen); // allocate memory on device
        runBsplineYdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, bc1y.get(), bc2y.get(), bc3y.get(), bc4y.get(), py.k0, py.b1, py.b2, py.norm_factor, boundary.get(), aStream);
    }
    if (flags & BSPLINE_X_DIR) {
        runBsplineXdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, bc1x.get(), bc2x.get(), bc3x.get(), bc4x.get(), px.k0, px.b1, px.b2, px.norm_factor, aStream);
    }
    if (flags & BSPLINE_Z_DIR) {
        runBsplineZdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, bc1z.get(), bc2z.get(), bc3z.get(), bc4z.get(), pz.k0, pz.b1, pz.b2, pz.norm_factor, aStream);
    }
    timer.stop_timer();
}

// explicit instantiation of handled types
template void cudaInverseBspline(PixelData<float> &, TypeOfInvBsplineFlags);
template <typename ImgType>
void cudaInverseBspline(PixelData<ImgType> &input, TypeOfInvBsplineFlags flags) {
    ScopedCudaMemHandler<PixelData<ImgType>, H2D | D2H> cudaInput(input);

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
    ScopedCudaMemHandler<const PixelData<ImageType>, H2D> cudaGrad(grad_temp);
    ScopedCudaMemHandler<PixelData<float>, D2H | H2D> cudaLis(local_scale_temp);

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
    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaImage(image);
    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaGrad(grad_temp);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudalocal_scale_temp(local_scale_temp);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudalocal_scale_temp2(local_scale_temp2);

    float tolerance = 0.0001;
    BsplineParams px = prepareBsplineStuff(image.x_num, par.lambda, tolerance);
    ScopedCudaMemHandler<float*, H2D> bc1x(px.bc1.get(), px.k0);
    ScopedCudaMemHandler<float*, H2D> bc2x(px.bc2.get(), px.k0);
    ScopedCudaMemHandler<float*, H2D> bc3x(px.bc3.get(), px.k0);
    ScopedCudaMemHandler<float*, H2D> bc4x(px.bc4.get(), px.k0);
    BsplineParams py = prepareBsplineStuff(image.y_num, par.lambda, tolerance);
    ScopedCudaMemHandler<float*, H2D> bc1y(py.bc1.get(), py.k0);
    ScopedCudaMemHandler<float*, H2D> bc2y(py.bc2.get(), py.k0);
    ScopedCudaMemHandler<float*, H2D> bc3y(py.bc3.get(), py.k0);
    ScopedCudaMemHandler<float*, H2D> bc4y(py.bc4.get(), py.k0);
    BsplineParams pz = prepareBsplineStuff(image.z_num, par.lambda, tolerance);
    ScopedCudaMemHandler<float*, H2D> bc1z(pz.bc1.get(), pz.k0);
    ScopedCudaMemHandler<float*, H2D> bc2z(pz.bc2.get(), pz.k0);
    ScopedCudaMemHandler<float*, H2D> bc3z(pz.bc3.get(), pz.k0);
    ScopedCudaMemHandler<float*, H2D> bc4z(pz.bc4.get(), pz.k0);

    int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * image.x_num * image.z_num;
    ScopedCudaMemHandler<float*, JUST_ALLOC> boundary(nullptr, boundaryLen);


    getGradientCuda(image, local_scale_temp, cudaImage.get(), cudaGrad.get(), cudalocal_scale_temp.get(),
                    px, py, pz, bc1x.get(), bc2x.get(), bc3x.get(), bc4x.get(), bc1y.get(), bc2y.get(), bc3y.get(), bc4y.get(), bc1z.get(), bc2z.get(), bc3z.get(), bc4z.get(), boundary.get(),
                    bspline_offset, par, 0);
}

// explicit instantiation of handled types
template void thresholdImg(PixelData<float> &, const float);
template <typename T>
void thresholdImg(PixelData<T> &image, const float threshold) {
    ScopedCudaMemHandler<PixelData<T>, H2D | D2H> cudaImage(image);

    runThresholdImg(cudaImage.get(), image.x_num, image.y_num, image.z_num, threshold, 0);
}

// explicit instantiation of handled types
template void thresholdGradient(PixelData<float> &, const PixelData<float> &, const float);
template <typename T>
void thresholdGradient(PixelData<float> &output, const PixelData<T> &input, const float Ip_th) {
    ScopedCudaMemHandler<const PixelData<T>, H2D> cudaInput(input);
    ScopedCudaMemHandler<PixelData<float>, H2D | D2H> cudaOutput(output);

    runThreshold(cudaInput.get(), cudaOutput.get(), input.x_num, input.y_num, input.z_num, Ip_th, 0);
}

void cudaDownsampledGradient(PixelData<float> &input, PixelData<float> &grad, const float hx, const float hy, const float hz) {
    ScopedCudaMemHandler<PixelData<float>, H2D | D2H> cudaInput(input);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudaGrad(grad);

    runKernelGradient(cudaInput.get(), cudaGrad.get(), input.x_num, input.y_num, input.z_num, grad.x_num, grad.y_num, hx, hy, hz, 0);
}
