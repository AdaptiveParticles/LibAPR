#include <iostream>
#include <chrono>
#include <cstdint>
#include <algorithm>

#include <cuda_runtime.h>

#include "ComputeGradientCuda.hpp"

#include <cuda_runtime_api.h>

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
#include "findMinMax.cuh"

#include "data_structures/APR/access/GenInfoGpuAccess.cuh"


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

        // TODO: for lambda == 0 this function should return empty BsplineParams, for now changing lambda
        // to generate anything (if lambda would stay 0 we get huge vectors out of range).
        if (lambda == 0) lambda = 0.1;

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
  
        // std::cout << std::fixed << std::setprecision(9) << "GPU: xi=" << xi << " rho=" << rho << " omg=" << omg << " gamma=" << gamma << " b1=" << b1
        //           << " b2=" << b2 << " k0=" << k0 << " minLen=" << minLen << " norm_factor=" << norm_factor << " lambda=" << lambda << " tol=" << tol << std::endl;

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

    // TODO: (APRstreams) isErrorDetected should be handled differently, in current state it blocks streams from
    //       running in parallel
    if (par.lambda > 0) {
        isErrorDetected = false;
        isErrorDetectedCuda.copyH2D();

        if (image.y_num > 2) runBsplineYdir(cudaImage, image.getDimension(), py, boundary, isErrorDetectedCuda.get(), aStream);
        if (image.x_num > 2) runBsplineXdir(cudaImage, image.getDimension(), px, isErrorDetectedCuda.get(), aStream);
        if (image.z_num > 2) runBsplineZdir(cudaImage, image.getDimension(), pz, isErrorDetectedCuda.get(), aStream);

        isErrorDetectedCuda.copyD2H();
        checkCuda(cudaStreamSynchronize(aStream));
        if (isErrorDetected) {
            throw std::invalid_argument("integer under-/overflow encountered in CUDA bspline(XYZ)dir - "
                                        "try squashing the input image to a narrower range or use APRConverter<float>");
        }
    }
    runKernelGradient(cudaImage, cudaGrad, image.getDimension(), local_scale_temp.getDimension(), par.dx, par.dy, par.dz, aStream);
    runDownsampleMean(cudaImage, cudalocal_scale_temp, image.x_num, image.y_num, image.z_num, aStream);

    if (par.lambda > 0) {
        if (image.y_num > 2) runInvBsplineYdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
        if (image.x_num > 2) runInvBsplineXdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
        if (image.z_num > 2) runInvBsplineZdir(cudalocal_scale_temp, local_scale_temp.x_num, local_scale_temp.y_num, local_scale_temp.z_num, aStream);
    }
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

/**
 * Thresholds output basing on input values. When input is < thresholdLevel then output is set to 0 and is not changed otherwise.
 * @param input
 * @param output
 * @param length - len of input/output arrays
 * @param thresholdLevel
 */
template <typename T, typename S>
__global__ void thresholdOpen(const T *input, S *output, size_t length, float thresholdLevel) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < length) {
        if (input[idx] < thresholdLevel) { output[idx] = 0; }
    }
}

template <typename ImgType, typename T>
void runThresholdOpen(ImgType *cudaImage, T *cudaGrad, size_t x_num, size_t y_num, size_t z_num, float Ip_th, cudaStream_t aStream) {
    dim3 threadsPerBlock(64);
    dim3 numBlocks((x_num * y_num * z_num + threadsPerBlock.x - 1)/threadsPerBlock.x);
    thresholdOpen<<<numBlocks,threadsPerBlock, 0, aStream>>>(cudaImage, cudaGrad, x_num * y_num * z_num, Ip_th);
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

class CudaStream {
    cudaStream_t iStream;

    /**
     * @return newly created stream
     */
    cudaStream_t getStream() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return stream;
    }

public:
    CudaStream() {
        iStream = getStream();
    }

    ~CudaStream() {
        cudaStreamDestroy(iStream);
    }

    cudaStream_t get() const {
        return iStream;
    }
};

template <typename U>
template <typename ImgType>
class GpuProcessingTask<U>::GpuProcessingTaskImpl {

    CudaStream cudaStream;
    const cudaStream_t iStream;

    // input data
    const PixelData<ImgType> &iCpuImage;
    PixelData<float> &iCpuLevels;
    const APRParameters &iParameters;
    GenInfo iAprInfo;
    float iBsplineOffset = 0;
    int iMaxLevel;

    // cuda stuff - memory and stream to be used
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


    // bool isErrorDetected;
    VectorData<bool> isErrorDetectedPinned;
    ScopedCudaMemHandler<bool *, JUST_ALLOC> isErrorDetectedCuda;

    const size_t boundaryLen;
    ScopedCudaMemHandler<float*, JUST_ALLOC> boundary;

    ParticleCellTreeCuda pctc;

    ScopedCudaMemHandler<uint16_t*, JUST_ALLOC> y_vec_cuda; // for LinearAccess
    LinearAccessCudaStructs lacs;

    // Padded memory for local_scale_temp and local_scale_temp2
    ScopedCudaMemHandler<float*, JUST_ALLOC> lstPadded;
    ScopedCudaMemHandler<float*, JUST_ALLOC> lst2Padded;


    // Structures used by computeLinearStructureCuda
    VectorData<uint64_t> xz_end_vec;
    VectorData<uint64_t> level_xz_vec;
    VectorData<uint16_t> y_vec;
    ScopedCudaMemHandler<uint64_t *, JUST_ALLOC> xz_end_vec_cuda; //(xz_end_vec.data(), xz_end_vec.size(), aStream);
    ScopedCudaMemHandler<uint64_t *, JUST_ALLOC> level_xz_vec_cuda; //(level_xz_vec.data(), level_xz_vec.size(), aStream);
    GenInfoGpuAccess giga;
    uint64_t counter_total = 1;

public:

    // TODO: Remove need for passing 'levels' to GpuProcessingTask
    //       It was used during development to control internal computation like filters, gradient, levels etc. but
    //       once all is done there is no need for it anymore
    GpuProcessingTaskImpl(const PixelData<ImgType> &inputImage, PixelData<float> &levels, const APRParameters &parameters, int maxLevel) :
        iCpuImage(inputImage),
        iCpuLevels(levels),
        iStream(cudaStream.get()),
        image (inputImage, iStream),
        gradient (levels, iStream),
        local_scale_temp (levels, iStream),
        local_scale_temp2 (levels, iStream),
        iParameters(parameters),
        iAprInfo(iCpuImage.getDimension()),
        iMaxLevel(maxLevel),
        cudax(transferSpline(prepareBsplineStuff(iCpuImage.x_num, iParameters.lambda, tolerance), iStream)),
        cuday(transferSpline(prepareBsplineStuff(iCpuImage.y_num, iParameters.lambda, tolerance), iStream)),
        cudaz(transferSpline(prepareBsplineStuff(iCpuImage.z_num, iParameters.lambda, tolerance), iStream)),
        isErrorDetectedPinned(true),
        isErrorDetectedCuda(nullptr, 1, iStream),
        boundaryLen{(2 /*two first elements*/ + 2 /* two last elements */) * (size_t)inputImage.x_num * (size_t)inputImage.z_num},
        boundary{nullptr, boundaryLen, iStream},
        pctc(iAprInfo, iStream),
        y_vec_cuda(nullptr, iAprInfo.getSize(), iStream),
        xz_end_vec(true),
        level_xz_vec(true),
        y_vec(true),
        giga(iAprInfo, iStream)
    {
        splineCudaX = cudax.first;
        splineCudaY = cuday.first;
        splineCudaZ = cudaz.first;
        // std::cout << "\n=============== GpuProcessingTaskImpl ===================" << iStream << "\n\n";
//        std::cout << iCpuImage << std::endl;
//        std::cout << iCpuLevels << std::endl;

        // In LIS we have: var_win[0,1,2] = maximum 3 var_win[3,4,5] = maximum 6
        // so maximum paddSize is 6 6 6
        PixelDataDim maxPaddSize(6, 6, 6);
        PixelDataDim paddedImageSize = levels.getDimension() + maxPaddSize + maxPaddSize;
        lstPadded.initialize(nullptr, paddedImageSize.size(), iStream);
        lst2Padded.initialize(nullptr, paddedImageSize.size(), iStream);


        // initialize_xz_linear() - CPU impl.
        counter_total = 1; //the buffer val to allow -1 calls without checking.
        level_xz_vec.resize(iAprInfo.l_max + 2, 0); //includes a buffer for -1 calls, and therefore needs to be called with level + 1;
        level_xz_vec[0] = 1; //allowing for the offset.
        for (int i = 0; i <= iAprInfo.l_max; ++i) {
            counter_total += iAprInfo.x_num[i] * iAprInfo.z_num[i];
            level_xz_vec[i + 1] = counter_total;
        }
        xz_end_vec.resize(counter_total, 0);
    // std::cout << "----------- iAprInfo.getSize() = " << iAprInfo.getSize() << std::endl;
        y_vec.resize(iAprInfo.getSize()); // resize it to worst case -> same number particles as pixels in input image
        // std::cout << "----------- iAprInfo.getSize() = " << iAprInfo.getSize() << std::endl;
        xz_end_vec_cuda.initialize(xz_end_vec.data(), xz_end_vec.size(), iStream);
        level_xz_vec_cuda.initialize(level_xz_vec.data(), level_xz_vec.size(), iStream);

        isErrorDetectedPinned.resize(1);
        isErrorDetectedCuda.initialize(isErrorDetectedPinned.data(), 1, iStream);
    }

    LinearAccessCudaStructs getDataFromGpu() {
        return std::move(lacs);
    }

    void processOnGpu() {
        // Set it and copy first before copying the image
        // It improves *a lot* performance even though it is needed later in computeLinearStructureCuda()
        iAprInfo.total_number_particles = 0; // reset total_number_particles to 0
        giga.copyHtoD();
        level_xz_vec_cuda.copyH2D();

        image.copyH2D();

        getGradientCuda(iCpuImage, iCpuLevels, image.get(), gradient.get(), local_scale_temp.get(),
                         splineCudaX, splineCudaY, splineCudaZ, boundary.get(), isErrorDetectedPinned[0], isErrorDetectedCuda,
                        iBsplineOffset, iParameters, iStream);

        runLocalIntensityScalePipeline(iCpuLevels, iParameters, local_scale_temp.get(), local_scale_temp2.get(), lstPadded.get(), lst2Padded.get(), iStream);

        // Apply parameters from APRConverter:
        runThreshold(local_scale_temp2.get(), gradient.get(), iCpuLevels.x_num, iCpuLevels.y_num, iCpuLevels.z_num, iParameters.Ip_th + iBsplineOffset, iStream);
        runRescaleAndThreshold(local_scale_temp.get(), iCpuLevels.mesh.size(), iParameters.sigma_th, iParameters.sigma_th_max, iStream);
        runThresholdOpen(gradient.get(), gradient.get(), iCpuLevels.x_num, iCpuLevels.y_num, iCpuLevels.z_num, iParameters.grad_th, iStream);
        // TODO: automatic parameters are not implemented for GPU pipeline (yet)

        float min_dim = std::min(iParameters.dy, std::min(iParameters.dx, iParameters.dz));
        float level_factor = pow(2, iMaxLevel) * min_dim;
        const float mult_const = level_factor/iParameters.rel_error;
        runComputeLevels(gradient.get(), local_scale_temp.get(), iCpuLevels.mesh.size(), mult_const, iStream);
        computeOvpcCuda(local_scale_temp.get(), pctc, iAprInfo, iStream);

        computeLinearStructureCuda(y_vec_cuda.get(), xz_end_vec_cuda.get(), level_xz_vec_cuda.get(), pctc, iAprInfo, giga, iParameters, counter_total, iStream);

        // Get data from GPU - first we need to get number of particles to resize y_vec and have idea how many particles to copy - that is why we need to synchronize first time
        giga.copyDtoH();
        checkCuda(cudaStreamSynchronize(iStream));

        // Start copying the data from GPU to CPU
        xz_end_vec_cuda.copyD2H();
        // Trim buffer to calculated size (initially it is allocated to worst case - same number of particles as pixels in input image) and copy data from GPU
        y_vec.resize(iAprInfo.total_number_particles);
        // Copy y_vec from GPU to CPU and synchronize last time - it is needed before we copy data to CPU structures
        checkCuda(cudaMemcpyAsync(y_vec.begin(), y_vec_cuda.get(), iAprInfo.total_number_particles * sizeof(uint16_t), cudaMemcpyDeviceToHost, iStream));

        // Synchornize last time - at that moment all data from GPU is copied to CPU
        checkCuda(cudaStreamSynchronize(iStream));

        // Prepare CPU structures
        lacs.xz_end_vec.copy(xz_end_vec);
        lacs.level_xz_vec.copy(level_xz_vec);
        lacs.y_vec.copy(y_vec);
    }

    void setBsplineOffset(float offset) {iBsplineOffset = offset;}

    ~GpuProcessingTaskImpl() {}
};

template <typename ImgType>
GpuProcessingTask<ImgType>::GpuProcessingTask(const PixelData<ImgType> &image, PixelData<float> &levels, const APRParameters &parameters, int maxLevel)
: impl{new GpuProcessingTaskImpl<ImgType>(image, levels, parameters, maxLevel)} { }

template <typename ImgType>
GpuProcessingTask<ImgType>::~GpuProcessingTask() { }

template <typename ImgType>
GpuProcessingTask<ImgType>::GpuProcessingTask(GpuProcessingTask&&) = default;

template <typename ImgType>
LinearAccessCudaStructs GpuProcessingTask<ImgType>::getDataFromGpu() {return impl->getDataFromGpu();}

template <typename ImgType>
void GpuProcessingTask<ImgType>::processOnGpu() {impl->processOnGpu();}

template <typename ImgType>
void GpuProcessingTask<ImgType>::setBsplineOffset(float offset) {impl->setBsplineOffset(offset);}

// explicit instantiation of handled types
template class GpuProcessingTask<uint8_t>;
template class GpuProcessingTask<int>;
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
        runBsplineZdir(cudaInput.get(), input.getDimension(), splineCuda, error.get(), aStream);
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


template<typename T>
std::pair<T,T> cudaRunMinMax(PixelData<T> &input_image) {
    cudaStream_t  aStream = nullptr;

    // Copy CPU image to CUDA mem
    ScopedCudaMemHandler<PixelData<T>, H2D> cudaImage(input_image, aStream);

    // In nvidia GPUs maximum number of threads per SM is multiplication of 512 (usually 1536 or 2048)
    // Calculate number of blocks to saturate whole SMs
    // Multiply it by 8 to have more smaller blocks to have better load balancing in case GPU is busy with other tasks
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int smCount = deviceProp.multiProcessorCount;
    const int numOfThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    constexpr int numOfThreads = 512;
    const int numOfBlocksPerSM = numOfThreadsPerSM / 512;
    const int maxNumberOfBlocks = smCount * numOfBlocksPerSM * 8;
    const size_t numOfElements = input_image.getDimension().size();
    int numOfBlocks = std::min(maxNumberOfBlocks, static_cast<int>((numOfElements + numOfThreads -1) / numOfThreads) );

    // Allocate memory for results both for CPU and GPU
    VectorData<T> minVector(true);
    VectorData<T> maxVector(true);
    minVector.resize(numOfBlocks);
    maxVector.resize(numOfBlocks);
    ScopedCudaMemHandler<T*, JUST_ALLOC> resultsMin(minVector.data(), numOfBlocks, aStream);
    ScopedCudaMemHandler<T*, JUST_ALLOC> resultsMax(maxVector.data(), numOfBlocks, aStream);

    // Run kernel and copy data back to CPU
    runFindMinMax(cudaImage.get(), input_image.getDimension(), aStream, resultsMin.get(), resultsMax.get(), numOfBlocks, numOfThreads);
    resultsMin.copyD2H();
    resultsMax.copyD2H();
    waitForCuda();

    // First values of minVector and maxVector contain min and max of all data
    return std::pair<T, T>(minVector[0], maxVector[0]);
}

template std::pair<uint16_t, uint16_t> cudaRunMinMax(PixelData<uint16_t> &);
template std::pair<int, int> cudaRunMinMax(PixelData<int> &);

