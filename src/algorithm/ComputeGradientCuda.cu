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
#include "algorithm/ParticleCellTreeCuda.cuh"
#include "algorithm/PullingSchemeCuda.hpp"
#include "data_structures/APR/access/LinearAccessCuda.hpp"

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
  
//        std::cout << std::fixed << std::setprecision(9) << "GPU: xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1
//                  << " b2=" << b2 << " k0=" << k0 << " minLen=" << minLen << " norm_factor=" << norm_factor << " lambda=" << lambda << " tol=" << tol << std::endl;

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

    auto transferSpline(const BsplineParams &aParams, cudaStream_t aStream) {
        ScopedCudaMemHandler<float*, H2D> bc1(aParams.bc1.get(), aParams.k0, aStream);
        ScopedCudaMemHandler<float*, H2D> bc2(aParams.bc2.get(), aParams.k0, aStream);
        ScopedCudaMemHandler<float*, H2D> bc3(aParams.bc3.get(), aParams.k0, aStream);
        ScopedCudaMemHandler<float*, H2D> bc4(aParams.bc4.get(), aParams.k0, aStream);

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
                     bool &isErrorDetected, ScopedCudaMemHandler<bool *, JUST_ALLOC>& isErrorDetectedCuda,
                     float bspline_offset, const APRParameters &par, cudaStream_t aStream) {

    // TODO: Used PixelDataDim in all methods below and change input parameter from image to imageDim

    isErrorDetected = false;
    isErrorDetectedCuda.copyH2D();
    if (image.y_num > 2) runBsplineYdir(cudaImage, image.getDimension(), py, boundary, isErrorDetectedCuda.get(), aStream);
    if (image.x_num > 2) runBsplineXdir(cudaImage, image.getDimension(), px, isErrorDetectedCuda.get(), aStream);
    // if (image.z_num > 2) runBsplineZdir(cudaImage, image.getDimension(), pz, aStream);
    // isErrorDetectedCuda.copyD2H();
    // if (isErrorDetected) {
    //     throw std::invalid_argument("integer under-/overflow encountered in CUDA bspline(XYZ)dir - "
    //                                 "try squashing the input image to a narrower range or use APRConverter<float>");
    // }
    //
    //
    // runKernelGradient(cudaImage, cudaGrad, image.getDimension(), local_scale_temp.getDimension(), par.dx, par.dy, par.dz, aStream);
    //
    // runDownsampleMean(cudaImage, cudalocal_scale_temp, image.x_num, image.y_num, image.z_num, aStream);
    //
    // if (image.y_num > 2) runInvBsplineYdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    // if (image.x_num > 2) runInvBsplineXdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    // if (image.z_num > 2) runInvBsplineZdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
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

template<typename T>
__global__ void rescaleAndThreshold(T *data, size_t len, float sigmaThreshold, float sigmaThresholdMax) {
    const float max_th = 60000.0;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float rescaled = data[idx];
        if (rescaled < sigmaThreshold) {
            rescaled = (rescaled < sigmaThresholdMax) ? max_th : sigmaThreshold;
        }
        data[idx] = rescaled;
    }
}

template <typename T>
void runRescaleAndThreshold(T *data, size_t len, float sigma, float sigmaMax, cudaStream_t aStream) {
    dim3 threadsPerBlock(64);
    dim3 numBlocks((len + threadsPerBlock.x - 1) / threadsPerBlock.x);
    rescaleAndThreshold <<< numBlocks, threadsPerBlock, 0, aStream >>> (data, len, sigma, sigmaMax);
}


template <typename U>
template <typename ImgType>
class GpuProcessingTask<U>::GpuProcessingTaskImpl {

    // input data
    const PixelData<ImgType> &iCpuImage;
    PixelData<float> &iCpuLevels;
    const APRParameters &iParameters;
    GenInfo iAprInfo;
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
    std::pair<BsplineParamsCuda, BsplineParamsCudaMemoryHandlers> cudax;
    std::pair<BsplineParamsCuda, BsplineParamsCudaMemoryHandlers> cuday;
    std::pair<BsplineParamsCuda, BsplineParamsCudaMemoryHandlers> cudaz;
    BsplineParamsCuda splineCudaX;
    BsplineParamsCuda splineCudaY;
    BsplineParamsCuda splineCudaZ;
    bool isErrorDetected;
    ScopedCudaMemHandler<bool *, JUST_ALLOC> isErrorDetectedCuda;

    const size_t boundaryLen;
    ScopedCudaMemHandler<float*, JUST_ALLOC> boundary;

    ParticleCellTreeCuda pctc;

    ScopedCudaMemHandler<uint16_t*, JUST_ALLOC> y_vec; // for LinearAccess
    LinearAccessCudaStructs lacs;

    /**
     * @return newly created stream
     */
    cudaStream_t getStream() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return stream;
    }

public:

    // TODO: Remove need for passing 'levels' to GpuProcessingTask
    //       It was used during development to control internal computation like filters, gradient, levels etc. but
    //       once all is done there is no need for it anymore
    GpuProcessingTaskImpl(const PixelData<ImgType> &inputImage, PixelData<float> &levels, const APRParameters &parameters, float bspline_offset, int maxLevel) :
        iCpuImage(inputImage),
        iCpuLevels(levels),
        iStream(getStream()),
        image (inputImage, iStream),
        gradient (levels, iStream),
        local_scale_temp (levels, iStream),
        local_scale_temp2 (levels, iStream),
        iParameters(parameters),
        iAprInfo(iCpuImage.getDimension()),
        iBsplineOffset(bspline_offset),
        iMaxLevel(maxLevel),
        cudax(transferSpline(prepareBsplineStuff(iCpuImage.x_num, iParameters.lambda, tolerance), iStream)),
        cuday(transferSpline(prepareBsplineStuff(iCpuImage.y_num, iParameters.lambda, tolerance), iStream)),
        cudaz(transferSpline(prepareBsplineStuff(iCpuImage.z_num, iParameters.lambda, tolerance), iStream)),
        isErrorDetectedCuda(&isErrorDetected, 1, iStream),
        boundaryLen{(2 /*two first elements*/ + 2 /* two last elements */) * (size_t)inputImage.x_num * (size_t)inputImage.z_num},
        boundary{nullptr, boundaryLen, iStream},
        pctc(iAprInfo, iStream),
        y_vec(nullptr, iAprInfo.getSize(), iStream)
    {
        splineCudaX = cudax.first;
        splineCudaY = cuday.first;
        splineCudaZ = cudaz.first;
        std::cout << "\n=============== GpuProcessingTaskImpl ===================" << iStream << "\n\n";
//        std::cout << iCpuImage << std::endl;
//        std::cout << iCpuLevels << std::endl;
    }

    void sendDataToGpu() {
//        CurrentTime ct;
//        uint64_t start = ct.microseconds();
        image.copyH2D();
//        checkCuda(cudaStreamSynchronize(iStream));
//        std::cout << "SEND time: " << ct.microseconds() - start << std::endl;
    }

    LinearAccessCudaStructs getDataFromGpu() {
        // TODO: Temporarily turned off here since synchronized already in computeLinearStructureCuda 
        // checkCuda(cudaStreamSynchronize(iStream));

        return std::move(lacs);
    }

    void processOnGpu() {
        // image.copyH2D();
        CurrentTime ct{};
        uint64_t start = ct.microseconds();

        CudaTimer time(false, "PIPELINE");
        time.start_timer("getgradient");
        getGradientCuda(iCpuImage, iCpuLevels, image.get(), gradient.get(), local_scale_temp.get(),
                         splineCudaX, splineCudaY, splineCudaZ, boundary.get(), isErrorDetected, isErrorDetectedCuda,
                        iBsplineOffset, iParameters, iStream);
        time.stop_timer();
        // time.start_timer("intensity");
        // runLocalIntensityScalePipeline(iCpuLevels, iParameters, local_scale_temp.get(), local_scale_temp2.get(), iStream);
        // time.stop_timer();
        //
        //
        // // Apply parameters from APRConverter:
        // time.start_timer("runs....");
        // runThreshold(local_scale_temp2.get(), gradient.get(), iCpuLevels.x_num, iCpuLevels.y_num, iCpuLevels.z_num, iParameters.Ip_th + iBsplineOffset, iStream);
        // runRescaleAndThreshold(local_scale_temp.get(), iCpuLevels.mesh.size(), iParameters.sigma_th, iParameters.sigma_th_max, iStream);
        // runThreshold(gradient.get(), gradient.get(), iCpuLevels.x_num, iCpuLevels.y_num, iCpuLevels.z_num, iParameters.grad_th, iStream);
        // // TODO: automatic parameters are not implemented for GPU pipeline (yet)
        // time.stop_timer();
        //
        // time.start_timer("compute lev");
        // float min_dim = std::min(iParameters.dy, std::min(iParameters.dx, iParameters.dz));
        // float level_factor = pow(2, iMaxLevel) * min_dim;
        // const float mult_const = level_factor/iParameters.rel_error;
        // runComputeLevels(gradient.get(), local_scale_temp.get(), iCpuLevels.mesh.size(), mult_const, iStream);
        // time.stop_timer();
        // computeOvpcCuda(local_scale_temp.get(), pctc, iAprInfo, iStream);
        // computeLinearStructureCuda(y_vec.get(), pctc, iAprInfo, iParameters, lacs, iStream);
    }

    ~GpuProcessingTaskImpl() {
        cudaStreamDestroy(iStream);
//        std::cout << "\n============== ~GpuProcessingTaskImpl ===================\n\n";
    }
};

template <typename ImgType>
GpuProcessingTask<ImgType>::GpuProcessingTask(const PixelData<ImgType> &image, PixelData<float> &levels, const APRParameters &parameters, float bspline_offset, int maxLevel)
: impl{new GpuProcessingTaskImpl<ImgType>(image, levels, parameters, bspline_offset, maxLevel)} { }

template <typename ImgType>
GpuProcessingTask<ImgType>::~GpuProcessingTask() { }

template <typename ImgType>
GpuProcessingTask<ImgType>::GpuProcessingTask(GpuProcessingTask&&) = default;

template <typename ImgType>
void GpuProcessingTask<ImgType>::sendDataToGpu() {impl->sendDataToGpu();}

template <typename ImgType>
LinearAccessCudaStructs GpuProcessingTask<ImgType>::getDataFromGpu() {return impl->getDataFromGpu();}

template <typename ImgType>
void GpuProcessingTask<ImgType>::processOnGpu() {impl->processOnGpu();}

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

    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaInput(input, aStream);

    APRTimer timer(false);
    bool isErrorDetected = false;
    ScopedCudaMemHandler<bool*, H2D | D2H> error(&isErrorDetected, 1, aStream);
    timer.start_timer("GpuDeviceTimeFull");
    if (flags & BSPLINE_Y_DIR) {
        BsplineParams p = prepareBsplineStuff((size_t)input.y_num, lambda, tolerance, maxFilterLen);
        auto cuda = transferSpline(p, aStream);
        auto splineCuda = cuda.first;
        int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * input.x_num * input.z_num;
        ScopedCudaMemHandler<float*, JUST_ALLOC> boundary(nullptr, boundaryLen, aStream); // allocate memory on device
        runBsplineYdir(cudaInput.get(), input.getDimension(), splineCuda, boundary.get(), error.get(), aStream);
    }
    if (flags & BSPLINE_X_DIR) {
        BsplineParams p = prepareBsplineStuff((size_t)input.x_num, lambda, tolerance, maxFilterLen);
        auto cuda = transferSpline(p, aStream);
        auto splineCuda = cuda.first;
        runBsplineXdir(cudaInput.get(), input.getDimension(), splineCuda, error.get(), aStream);
    }
    if (flags & BSPLINE_Z_DIR) {
        BsplineParams p = prepareBsplineStuff((size_t)input.z_num, lambda, tolerance, maxFilterLen);
        auto cuda = transferSpline(p, aStream);
        auto splineCuda = cuda.first;
        runBsplineZdir(cudaInput.get(), input.getDimension(), splineCuda, aStream);
    }

    waitForCuda();

    if (isErrorDetected) {
        throw std::invalid_argument("integer under-/overflow encountered in CUDA bspline(XYZ)dir - "
                                    "try squashing the input image to a narrower range or use APRConverter<float>");
    }

    timer.stop_timer();
}

// explicit instantiation of handled types
template void cudaInverseBspline(PixelData<float> &, TypeOfInvBsplineFlags);
template <typename ImgType>
void cudaInverseBspline(PixelData<ImgType> &input, TypeOfInvBsplineFlags flags) {
    cudaStream_t  aStream = 0;

    ScopedCudaMemHandler<PixelData<ImgType>, H2D | D2H> cudaInput(input, aStream);

    if (flags & INV_BSPLINE_Y_DIR) {
        runInvBsplineYdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, aStream);
    }
    if (flags & INV_BSPLINE_X_DIR) {
        runInvBsplineXdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, aStream);
    }
    if (flags & INV_BSPLINE_Z_DIR) {
        runInvBsplineZdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, aStream);
    }
}

// explicit instantiation of handled types
template void computeLevelsCuda(const PixelData<float> &, PixelData<float> &, int, float, float, float, float);
template <typename ImageType>
void computeLevelsCuda(const PixelData<ImageType> &grad_temp, PixelData<float> &local_scale_temp, int maxLevel, float relError,  float dx, float dy, float dz) {
    cudaStream_t  aStream = 0;

    ScopedCudaMemHandler<const PixelData<ImageType>, H2D> cudaGrad(grad_temp, aStream);
    ScopedCudaMemHandler<PixelData<float>, D2H | H2D> cudaLis(local_scale_temp, aStream);

    float min_dim = std::min(dy, std::min(dx, dz));
    float level_factor = pow(2, maxLevel) * min_dim;
    const float mult_const = level_factor/relError;
    runComputeLevels(cudaGrad.get(), cudaLis.get(), grad_temp.mesh.size(), mult_const, aStream);
}

// explicit instantiation of handled types
template void getGradient(PixelData<float> &, PixelData<float> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &);
template void getGradient(PixelData<uint16_t> &, PixelData<uint16_t> &, PixelData<float> &, PixelData<float> &, float, const APRParameters &);

template <typename ImgType>
void getGradient(PixelData<ImgType> &image, PixelData<ImgType> &grad_temp, PixelData<float> &local_scale_temp, PixelData<float> &local_scale_temp2, float bspline_offset, const APRParameters &par) {
    cudaStream_t  aStream = 0;
    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaImage(image, aStream);
    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaGrad(grad_temp, aStream);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudalocal_scale_temp(local_scale_temp, aStream);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudalocal_scale_temp2(local_scale_temp2, aStream);

    int boundaryLen = (2 /*two first elements*/ + 2 /* two last elements */) * image.x_num * image.z_num;
    ScopedCudaMemHandler<float*, JUST_ALLOC> boundary(nullptr, boundaryLen, aStream);

    float tolerance = 0.0001;

    // TODO: This is wrong and done only for compile. BsplineParams has to be computed seperately for each dimension.
    //       Should be fixed when other parts of pipeline are ready.

    // FIX BSPLINE PARAMS !!!!!!!! to get full gradient pipeline test working !!!!!!!!!!!!!!!!!!!!!!!!!1


    BsplineParams px = prepareBsplineStuff(image.x_num, par.lambda, tolerance);
    auto cudax = transferSpline(px, aStream);
    auto splineCudaX = cudax.first;
    BsplineParams py = prepareBsplineStuff(image.y_num, par.lambda, tolerance);
    auto cuday = transferSpline(py, aStream);
    auto splineCudaY = cuday.first;
    BsplineParams pz = prepareBsplineStuff(image.z_num, par.lambda, tolerance);
    auto cudaz = transferSpline(pz, aStream);
    auto splineCudaZ = cudaz.first;
    bool isErrorDetected = false;
    {
        ScopedCudaMemHandler<bool*, JUST_ALLOC> isErrorDetectedCuda(&isErrorDetected, 1, aStream);
        getGradientCuda(image, local_scale_temp, cudaImage.get(), cudaGrad.get(), cudalocal_scale_temp.get(),
                        splineCudaX, splineCudaY, splineCudaZ, boundary.get(), isErrorDetected, isErrorDetectedCuda, bspline_offset, par, aStream);
    }
}

void cudaDownsampledGradient(PixelData<float> &input, PixelData<float> &grad, const float hx, const float hy, const float hz) {
    cudaStream_t  aStream = 0;

    ScopedCudaMemHandler<PixelData<float>, H2D | D2H> cudaInput(input, aStream);
    ScopedCudaMemHandler<PixelData<float>, D2H> cudaGrad(grad, aStream);

    runKernelGradient(cudaInput.get(), cudaGrad.get(), input.getDimension(), grad.getDimension(), hx, hy, hz, aStream);
}
