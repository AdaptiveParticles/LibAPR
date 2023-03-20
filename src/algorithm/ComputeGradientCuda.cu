#include <iostream>
#include <chrono>
#include <cstdint>
#include <algorithm>

#include <cuda_runtime.h>

#include "ComputeGradientCuda.hpp"
#include "APRParameters.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/Mesh/downsample.cuh"
#include "algorithm/ComputePullingScheme.cuh"
#include "algorithm/LocalIntensityScale.cuh"
#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"

#include "dsGradient.cuh"
#include "invBspline.cuh"
#include "bsplineParams.h"
#include "bsplineXdir.cuh"
#include "bsplineYdir.cuh"
#include "bsplineZdir.cuh"



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

    struct BsplineParamsCudaMemoryHandlers {
        ScopedCudaMemHandler<float*, H2D> bc1;
        ScopedCudaMemHandler<float*, H2D> bc2;
        ScopedCudaMemHandler<float*, H2D> bc3;
        ScopedCudaMemHandler<float*, H2D> bc4;
    };

    float impulse_resp(float k, float rho, float omg) {
        //  Impulse Response Function
        return (powf(rho, (std::abs(k))) * sinf((std::abs(k) + 1) * omg)) / sinf(omg);
    }

    float impulse_resp_back(float k, float rho, float omg, float gamma, float c0) {
        //  Impulse Response Function (nominator eq. 4.8, denominator from eq. 4.7)
        return c0 * powf(rho, std::abs(k)) * (cosf(omg * std::abs(k)) + gamma * sinf(omg * std::abs(k))) *
               (1.0 / (powf((1 - 2.0 * rho * cosf(omg) + powf(rho, 2)), 2)));
    }

    BsplineParams prepareBsplineStuff(size_t dimLen, float lambda, float tol, int maxFilterLen = -1) {
        // Recursive Filter Implimentation for Smoothing BSplines
        // B-Spline Signal Processing: Part II - Efficient Design and Applications, Unser 1993

        float xi = 1 - 96 * lambda + 24 * lambda * sqrtf(3 + 144 * lambda); // eq 4.6
        float rho = (24 * lambda - 1 - sqrtf(xi)) / (24 * lambda) *
                    sqrtf((1 / xi) * (48 * lambda + 24 * lambda * sqrtf(3 + 144 * lambda))); // eq 4.5

        float omg = atan(sqrtf((1 / xi) * (144 * lambda - 1))); // eq 4.6

        float c0 = (1 + powf(rho, 2)) / (1 - powf(rho, 2)) * (1 - 2 * rho * cosf(omg) + powf(rho, 2)) /
                   (1 + 2 * rho * cosf(omg) + powf(rho, 2)); // eq 4.8
        float gamma = (1 - powf(rho, 2)) / (1 + powf(rho, 2)) * (1 / tan(omg)); // eq 4.8

        const float b1 = 2 * rho * cosf(omg);
        const float b2 = -powf(rho, 2.0);

        const size_t idealK0Len = ceil(std::abs(logf(tol) / logf(rho)));
        const size_t k0 = maxFilterLen > 0 ? maxFilterLen : idealK0Len;
        const size_t minLen = maxFilterLen > 0 ? maxFilterLen : std::min(idealK0Len, dimLen);

        const float norm_factor = powf((1 - 2.0 * rho * cosf(omg) + powf(rho, 2)), 2);
  
        //std::cout << std::fixed << std::setprecision(9) << "GPU: xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1
        //          << " b2=" << b2 << " k0=" << k0 << " minLen=" << minLen << " norm_factor=" << norm_factor << std::endl;

        // ------- Calculating boundary conditions

        size_t boundaryLen = sizeof(float) * k0;
        PinnedMemoryUniquePtr<float> bc1{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc2{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc3{(float*)getPinnedMemory(boundaryLen)};
        PinnedMemoryUniquePtr<float> bc4{(float*)getPinnedMemory(boundaryLen)};

        // forward boundaries
        std::vector<float> impulse_resp_vec_f(k0 + 1);
        for (size_t k = 0; k < impulse_resp_vec_f.size(); ++k) impulse_resp_vec_f[k] = impulse_resp(k, rho, omg);

        //y(0) init
        for (size_t k = 0; k < k0; ++k) bc1[k] = impulse_resp_vec_f[k];
        for (size_t k = minLen; k < k0; ++k) bc1[minLen - 1] += bc1[k];

        //y(1) init
        for (size_t k = 0; k < k0; ++k) bc2[k] = 0;
        bc2[1] = impulse_resp_vec_f[0];
        for (size_t k = 0; k < k0; ++k) bc2[k] += impulse_resp_vec_f[k + 1];
        for (size_t k = minLen; k < k0; ++k) bc2[minLen - 1] += bc2[k];

        // backward boundaries
        std::vector<float> impulse_resp_vec_b(k0 + 1);
        for (size_t k = 0; k < impulse_resp_vec_b.size(); ++k)
            impulse_resp_vec_b[k] = impulse_resp_back(k, rho, omg, gamma, c0);

        //y(N-1) init
        for (size_t k = 0; k < k0; ++k) bc3[k] = 0;
        bc3[0] = impulse_resp_vec_b[1];
        for (size_t k = 0; k < (k0 - 1); ++k) bc3[k + 1] += impulse_resp_vec_b[k] + impulse_resp_vec_b[k + 2];
        for (size_t k = minLen; k < k0; ++k) bc3[minLen - 1] += bc3[k];

        //y(N) init
        for (size_t k = 0; k < k0; ++k) bc4[k] = 0;
        bc4[0] = impulse_resp_vec_b[0];
        for (size_t k = 1; k < k0; ++k) bc4[k] += 2 * impulse_resp_vec_b[k];
        for (size_t k = minLen; k < k0; ++k) bc4[minLen - 1] += bc4[k];

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

    auto transferSpline(BsplineParams &aParams) {
        ScopedCudaMemHandler<float*, H2D> bc1(aParams.bc1.get(), aParams.k0);
        ScopedCudaMemHandler<float*, H2D> bc2(aParams.bc2.get(), aParams.k0);
        ScopedCudaMemHandler<float*, H2D> bc3(aParams.bc3.get(), aParams.k0);
        ScopedCudaMemHandler<float*, H2D> bc4(aParams.bc4.get(), aParams.k0);

        return std::pair<BsplineParamsCuda, BsplineParamsCudaMemoryHandlers> {
                BsplineParamsCuda {
                        bc1.get(),
                        bc2.get(),
                        bc3.get(),
                        bc4.get(),
                        aParams.k0,
                        aParams.b1,
                        aParams.b2,
                        aParams.norm_factor
                },

                BsplineParamsCudaMemoryHandlers {
                        std::move(bc1),
                        std::move(bc2),
                        std::move(bc3),
                        std::move(bc4)
                }
        };
    }
}

template <typename ImgType>
void getGradientCuda(const PixelData<ImgType> &image, PixelData<float> &local_scale_temp,
                     ImgType *cudaImage, ImgType *cudaGrad, float *cudalocal_scale_temp,
                     BsplineParamsCuda &px, BsplineParamsCuda &py, BsplineParamsCuda &pz, float *boundary,
                     float bspline_offset, const APRParameters &par, cudaStream_t aStream) {

    runBsplineYdir(cudaImage, image.getDimension(), py, boundary, aStream);
    runBsplineXdir(cudaImage, image.getDimension(), px, aStream);
    runBsplineZdir(cudaImage, image.getDimension(), pz, aStream);

    runKernelGradient(cudaImage, cudaGrad, image.getDimension(), local_scale_temp.getDimension(), par.dx, par.dy, par.dz, aStream);

    runDownsampleMean(cudaImage, cudalocal_scale_temp, image.x_num, image.y_num, image.z_num, aStream);

    runInvBsplineYdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    runInvBsplineXdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    runInvBsplineZdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
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
        // TODO: This is wrong and done only for compile. BsplineParams has to be computed seperately for each dimension.
        //       Should be fixed when other parts of pipeline are ready.
        params(prepareBsplineStuff((size_t)image.x_num, parameters.lambda, tolerance)),
        bc1(params.bc1.get(), params.k0, iStream),
        bc2(params.bc2.get(), params.k0, iStream),
        bc3(params.bc3.get(), params.k0, iStream),
        bc4(params.bc4.get(), params.k0, iStream),
        boundaryLen{(2 /*two first elements*/ + 2 /* two last elements */) * (size_t)image.x_num * (size_t)image.z_num},
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

        // TODO: temporarily bspline params are generated here
        //       In principle this is OK and correct but would be faster (for processing series of same size images) if
        //       they would be calculated in constructor of GpuProcessingTaskImpl class (once).
        BsplineParams px = prepareBsplineStuff(iCpuImage.x_num, iParameters.lambda, tolerance);
        auto cudax = transferSpline(px);
        auto splineCudaX = cudax.first;
        BsplineParams py = prepareBsplineStuff(iCpuImage.y_num, iParameters.lambda, tolerance);
        auto cuday = transferSpline(py);
        auto splineCudaY = cuday.first;
        BsplineParams pz = prepareBsplineStuff(iCpuImage.z_num, iParameters.lambda, tolerance);
        auto cudaz = transferSpline(pz);
        auto splineCudaZ = cudaz.first;

        getGradientCuda(iCpuImage, iCpuLevels, image.get(), gradient.get(), local_scale_temp.get(),
                         splineCudaX, splineCudaY, splineCudaZ, boundary.get(),
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
template void cudaFilterBsplineFull(PixelData<uint16_t> &, float, float, TypeOfRecBsplineFlags, int);
template void cudaFilterBsplineFull(PixelData<int16_t> &, float, float, TypeOfRecBsplineFlags, int);
template void cudaFilterBsplineFull(PixelData<uint8_t> &, float, float, TypeOfRecBsplineFlags, int);



template <typename ImgType>
void cudaFilterBsplineFull(PixelData<ImgType> &input, float lambda, float tolerance, TypeOfRecBsplineFlags flags, int maxFilterLen) {
    cudaStream_t  aStream = 0;


    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaInput(input);

    APRTimer timer(false);
    timer.start_timer("GpuDeviceTimeFull");
    if (flags & BSPLINE_Y_DIR) {
        BsplineParams p = prepareBsplineStuff((size_t)input.y_num, lambda, tolerance, maxFilterLen);
        auto cuda = transferSpline(p);
        auto splineCuda = cuda.first;
        int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
        ScopedCudaMemHandler<float*, JUST_ALLOC> boundary(nullptr, boundaryLen); // allocate memory on device
        runBsplineYdir(cudaInput.get(), input.getDimension(), splineCuda, boundary.get(), aStream);
    }
    if (flags & BSPLINE_X_DIR) {
        BsplineParams p = prepareBsplineStuff((size_t)input.x_num, lambda, tolerance, maxFilterLen);
        auto cuda = transferSpline(p);
        auto splineCuda = cuda.first;
        runBsplineXdir(cudaInput.get(), input.getDimension(), splineCuda, aStream);
    }
    if (flags & BSPLINE_Z_DIR) {
        BsplineParams p = prepareBsplineStuff((size_t)input.z_num, lambda, tolerance, maxFilterLen);
        auto cuda = transferSpline(p);
        auto splineCuda = cuda.first;
        runBsplineZdir(cudaInput.get(), input.getDimension(), splineCuda, aStream);
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
template void getGradient(PixelData<uint16_t> &, PixelData<uint16_t> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &);

template <typename ImgType>
void getGradient(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par) {
    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaImage(image);
    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaGrad(grad_temp);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudalocal_scale_temp(local_scale_temp);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudalocal_scale_temp2(local_scale_temp2);

    int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * image.x_num * image.z_num;
    ScopedCudaMemHandler<float*, JUST_ALLOC> boundary(nullptr, boundaryLen);

    float tolerance = 0.0001;


    // TODO: This is wrong and done only for compile. BsplineParams has to be computed seperately for each dimension.
    //       Should be fixed when other parts of pipeline are ready.

    // FIX BSPLINE PARAMS !!!!!!!! to get full gradient pipeline test working !!!!!!!!!!!!!!!!!!!!!!!!!1


    BsplineParams px = prepareBsplineStuff(image.x_num, par.lambda, tolerance);
    auto cudax = transferSpline(px);
    auto splineCudaX = cudax.first;
    BsplineParams py = prepareBsplineStuff(image.y_num, par.lambda, tolerance);
    auto cuday = transferSpline(py);
    auto splineCudaY = cuday.first;
    BsplineParams pz = prepareBsplineStuff(image.z_num, par.lambda, tolerance);
    auto cudaz = transferSpline(pz);
    auto splineCudaZ = cudaz.first;

    getGradientCuda(image, local_scale_temp, cudaImage.get(), cudaGrad.get(), cudalocal_scale_temp.get(),
                    splineCudaX, splineCudaY, splineCudaZ, boundary.get(), bspline_offset, par, 0);
}

void cudaDownsampledGradient(PixelData<float> &input, PixelData<float> &grad, const float hx, const float hy, const float hz) {
    ScopedCudaMemHandler<PixelData<float>, H2D | D2H> cudaInput(input);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudaGrad(grad);

    runKernelGradient(cudaInput.get(), cudaGrad.get(), input.getDimension(), grad.getDimension(), hx, hy, hz, 0);
}
