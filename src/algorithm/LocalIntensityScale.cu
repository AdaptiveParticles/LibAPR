#include "LocalIntensityScaleCuda.h"

#include "LocalIntensityScale.hpp"

#include <iostream>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <cuda_runtime_api.h>
//#include <cuda_runtime.h>

#include "misc/CudaTools.cuh"
#include "data_structures/Mesh/paddPixelData.cuh"

/**
 * Calculates mean in Y direction
 *
 * NOTE: This is not optimal implementation but.. correct and more or less fast as previous one.
 *       The reason for change was to have results exactly same as in CPU side.
 *       Currently after reading whole y-dir line of data mean calculation is done only by one from all threads in block
 *       so here is some room for improvements.
 *       If needed may be optimized in future. The main limitation is size of shared memory needed which
 *       limits number of CUDA blocks that can run in parallel.
 * @tparam T
 * @param image
 * @param offset
 * @param x_num
 * @param y_num
 * @param z_num
 */
template <typename T>
__global__ void meanYdir(T *image, int offset, size_t x_num, size_t y_num, size_t z_num, bool boundaryReflect) {
    // NOTE: Block size in x/z direction must be 1
    const size_t workersOffset = (blockIdx.z * x_num + blockIdx.x) * y_num;
    const int numOfWorkers = blockDim.y;
    const int workerIdx = threadIdx.y;

    extern __shared__ char sharedMemChar[];
    T *buffer = (T*) sharedMemChar;
    T *data = (T*) &buffer[y_num];

    // Read whole line of data from y-direction
    int workerOffset = workerIdx;
    while (workerOffset < y_num) {
        buffer[workerOffset] = image[workersOffset + workerOffset];
        workerOffset += numOfWorkers;
    }

    const int divisor = 2 * offset  + 1;
    size_t currElementOffset = 0;
    size_t saveElementOffset = 0;
    size_t nextElementOffset = 1;

    if (workerIdx == 0) {
        // clear shared mem
        for (int i = offset; i < divisor; ++i) data[i] = 0;

        // saturate cache with #offset elements since it will allow to calculate first element value on LHS
        float sum = 0;
        int count = 0;
        while (count <= offset) {
            T v = buffer[currElementOffset];
            sum += v;
            data[count] = v;
            if (boundaryReflect && count > 0) {
                data[2 * offset - count + 1] = v;
                sum += v;
            }
            currElementOffset += nextElementOffset;
            ++count;
        }

        if (boundaryReflect) {
            count += offset; // elements in above loop in range [1, offset] were summed twice
        }

        // Pointer in circular buffer
        int beginPtr = (offset + 1) % divisor;

        // main loop going through all elements in range [0, y_num - 1 - offset], so till last element that
        // does not need handling RHS for offset '^'
        // x x x x ... x x x x x x x
        //                 o o ^ o o
        //
        const int lastElement = y_num - 1 - offset;
        for (int y = 0; y <= lastElement; ++y) {
            // Calculate and save currently processed element and move to the new one
            buffer[saveElementOffset] = sum / count;
            saveElementOffset += nextElementOffset;

            // There is no more elements to process in that loop, all stuff left to be processed is already in 'data' buffer
            if (y == lastElement) break;

            // Read new element
            T v = buffer[currElementOffset];

            // Update sum to cover [-offset, offset] of currently processed element
            sum -= data[beginPtr];
            sum += v;

            // Store new element in circularBuffer
            data[beginPtr] = v;

            // Move to next elements to read and in circular buffer
            count = min(count + 1, divisor);
            beginPtr = (beginPtr + 1) % divisor;
            currElementOffset += nextElementOffset;
        }

        // Handle last #offset elements on RHS
        int boundaryPtr = (beginPtr - 1 - 1 + (2 * offset + 1)) % divisor;

        while (saveElementOffset < currElementOffset) {
            // If filter length is too big in comparison to processed dimension
            // do not decrease 'count' and do not remove first element from moving filter
            // since 'sum' of filter elements contains all elements from processed dimension:
            // dim elements:        xxxxxx
            // filter elements:  oooooo^ooooo   (o - offset elements, ^ - middle of the filter)
            // In such a case first 'o' element should not be removed when filter moves right.
            if (y_num - (currElementOffset - saveElementOffset) / nextElementOffset > offset || boundaryReflect) {
                if (!boundaryReflect) count = count - 1;
                sum -= data[beginPtr];
            }

            if (boundaryReflect) {
                sum += data[boundaryPtr];
                boundaryPtr = (boundaryPtr - 1 + (2 * offset + 1)) % divisor;
            }

            buffer[saveElementOffset] = sum / count;
            beginPtr = (beginPtr + 1) % divisor;
            saveElementOffset += nextElementOffset;
        }
    }

    // Save whole line of data
    workerOffset = workerIdx;
    while (workerOffset < y_num) {
        image[workersOffset + workerOffset] = buffer[workerOffset];
        workerOffset += numOfWorkers;
    }
}
constexpr int NumberOfWorkers = 32; // Cannot be greater than 32 since there is no inter-warp communication implemented.

/**
 * Filter in X-dir moves circular buffer along direction adding to sum of elements newly read element and removing last one.
 * For instance (filter len = 5)
 *
 * idx:               0 1 2 3 4 5 6 7 8 9
 * image elements:    1 2 2 4 5 3 2 1 3 4
 *
 * buffer:                2 3 4 5 2                        current sum = 16  element @idx=4 will be updated to 16/5
 *
 * next step
 * buffer:                  3 4 5 2 1                      sum = sum - 2 + 1 = 15  element @idx=5 = 15 / 5
 *
 * In general circular buffer is kept to speedup operations and to not reach to global memory more than once for
 * read/write operations for given element.
 */
template <typename T>
__global__ void meanXdir(T *image, int offset, size_t x_num, size_t y_num, size_t z_num, bool boundaryReflect = false) {
    const size_t workerOffset = blockIdx.y * blockDim.y + threadIdx.y + (blockIdx.z * blockDim.z + threadIdx.z) * y_num * x_num;
    const int workerYoffset = blockIdx.y * blockDim.y + threadIdx.y ;
    const int workerIdx = threadIdx.y;
    const int nextElementOffset = y_num;

    extern __shared__ float sharedMem[];
    float (*data)[NumberOfWorkers] = (float (*)[NumberOfWorkers])sharedMem;

    const int divisor = 2 * offset  + 1;
    size_t currElementOffset = 0;
    size_t saveElementOffset = 0;

    if (workerYoffset < y_num) {
        // clear shared mem
        for (int i = offset; i < divisor; ++i) data[i][workerIdx] = 0;

        // saturate cache with #offset elements since it will allow to calculate first element value on LHS
        float sum = 0;
        int count = 0;
        while (count <= offset) {
            T v = image[workerOffset + currElementOffset];
            sum += v;
            data[count][workerIdx] = v;
            if (boundaryReflect && count > 0) {data[2 * offset - count + 1][workerIdx] = v; sum += v;}
            currElementOffset += nextElementOffset;
            ++count;
        }

        if (boundaryReflect) {
            count += offset; // elements in above loop in range [1, offset] were summed twice
        }

        // Pointer in circular buffer
        int beginPtr = (offset + 1) % divisor;

        // main loop going through all elements in range [0, x_num - 1 - offset], so till last element that
        // does not need handling RHS for offset '^'
        // x x x x ... x x x x x x x
        //                 o o ^ o o
        //
        const int lastElement = x_num - 1 - offset;
        for (int x = 0; x <= lastElement; ++x) {
            // Calculate and save currently processed element and move to the new one
            image[workerOffset + saveElementOffset] = sum / count;
            saveElementOffset += nextElementOffset;

            // There is no more elements to process in that loop, all stuff left to be processed is already in 'data' buffer
            if (x == lastElement) break;

            // Read new element
            T v = image[workerOffset + currElementOffset];

            // Update sum to cover [-offset, offset] of currently processed element
            sum -= data[beginPtr][workerIdx];
            sum += v;

            // Store new element in circularBuffer
            data[beginPtr][workerIdx] = v;

            // Move to next elements to read and in circular buffer
            count = min(count + 1, divisor);
            beginPtr = (beginPtr + 1) % divisor;
            currElementOffset += nextElementOffset;
        }

        // Handle last #offset elements on RHS
        int boundaryPtr = (beginPtr - 1 - 1 + (2*offset+1)) % divisor;

        while (saveElementOffset < currElementOffset) {
            // If filter length is too big in comparison to processed dimension
            // do not decrease 'count' and do not remove first element from moving filter
            // since 'sum' of filter elements contains all elements from processed dimension:
            // dim elements:        xxxxxx
            // filter elements:  oooooo^ooooo   (o - offset elements, ^ - middle of the filter)
            // In such a case first 'o' element should not be removed when filter moves right.
            if (x_num - (currElementOffset - saveElementOffset)/nextElementOffset > offset || boundaryReflect) {
                if (!boundaryReflect) count = count - 1;
                sum -= data[beginPtr][workerIdx];
            }

            if (boundaryReflect) {
                sum += data[boundaryPtr][workerIdx];
                boundaryPtr = (boundaryPtr - 1 + (2*offset+1)) % divisor;
            }

            image[workerOffset + saveElementOffset] = sum / count;
            beginPtr = (beginPtr + 1) % divisor;
            saveElementOffset += nextElementOffset;
        }
    }
}

/**
 * Filter in Z-dir moves circular buffer along direction adding to sum of elements newly read element and removing last one.
 * For instance (filter len = 5)
 *
 * idx:               0 1 2 3 4 5 6 7 8 9
 * image elements:    1 2 2 4 5 3 2 1 3 4
 *
 * buffer:                2 3 4 5 2                        current sum = 16  element @idx=4 will be updated to 16/5
 *
 * next step
 * buffer:                  3 4 5 2 1                      sum = sum - 2 + 1 = 15  element @idx=5 = 15 / 5
 *
 * In general circular buffer is kept to speedup operations and to not reach to global memory more than once for
 * read/write operations for given element.
 */
template <typename T>
__global__ void meanZdir(T *image, int offset, size_t x_num, size_t y_num, size_t z_num, bool boundaryReflect = false) {
    const size_t workerOffset = blockIdx.y * blockDim.y + threadIdx.y + (blockIdx.z * blockDim.z + threadIdx.z) * y_num; // *.z is 'x'
    const int workerYoffset = blockIdx.y * blockDim.y + threadIdx.y ;
    const int workerIdx = threadIdx.y;
    const int nextElementOffset = x_num * y_num;

    extern __shared__ float sharedMem[];
    float (*data)[NumberOfWorkers] = (float (*)[NumberOfWorkers])sharedMem;

    const int divisor = 2 * offset  + 1;
    size_t currElementOffset = 0;
    size_t saveElementOffset = 0;

    if (workerYoffset < y_num) {
        // clear shared mem
        for (int i = offset; i < divisor; ++i) data[i][workerIdx] = 0;

        // saturate cache with #offset elements since it will allow to calculate first element value on LHS
        float sum = 0;
        int count = 0;
        while (count <= offset) {
            T v = image[workerOffset + currElementOffset];
            sum += v;
            data[count][workerIdx] = v;
            if (boundaryReflect && count > 0) {data[2 * offset - count + 1][workerIdx] = v; sum += v;}
            currElementOffset += nextElementOffset;
            ++count;
        }

        if (boundaryReflect) {
            count += offset; // elements in above loop in range [1, offset] were summed twice
        }

        // Pointer in circular buffer
        int beginPtr = (offset + 1) % divisor;

        // main loop going through all elements in range [0, z_num - 1 - offset], so till last element that
        // does not need handling RHS for offset '^'
        // x x x x ... x x x x x x x
        //                 o o ^ o o
        //
        const int lastElement = z_num - 1 - offset;
        for (int z = 0; z <= lastElement; ++z) {
            // Calculate and save currently processed element and move to the new one
            image[workerOffset + saveElementOffset] = sum / count;
            saveElementOffset += nextElementOffset;

            // There is no more elements to process in that loop, all stuff left to be processed is already in 'data' buffer
            if (z == lastElement) break;

            // Read new element
            T v = image[workerOffset + currElementOffset];

            // Update sum to cover [-offset, offset] of currently processed element
            sum -= data[beginPtr][workerIdx];
            sum += v;

            // Store new element in circularBuffer
            data[beginPtr][workerIdx] = v;

            // Move to next elements to read and in circular buffer
            count = min(count + 1, divisor);
            beginPtr = (beginPtr + 1) % divisor;
            currElementOffset += nextElementOffset;
        }

        // Handle last #offset elements on RHS
        int boundaryPtr = (beginPtr - 1 - 1 + (2*offset+1)) % divisor;

        while (saveElementOffset < currElementOffset) {
            // If filter length is too big in comparison to processed dimension
            // do not decrease 'count' and do not remove first element from moving filter
            // since 'sum' of filter elements contains all elements from processed dimension:
            // dim elements:        xxxxxx
            // filter elements:  oooooo^ooooo   (o - offset elements, ^ - middle of the filter)
            // In such a case first 'o' element should not be removed when filter moves right.
            if (z_num - (currElementOffset - saveElementOffset)/nextElementOffset > offset || boundaryReflect) {
                if (!boundaryReflect) count = count - 1;
                sum -= data[beginPtr][workerIdx];
            }

            if (boundaryReflect) {
                sum += data[boundaryPtr][workerIdx];
                boundaryPtr = (boundaryPtr - 1 + (2*offset+1)) % divisor;
            }

            image[workerOffset + saveElementOffset] = sum / count;
            beginPtr = (beginPtr + 1) % divisor;
            saveElementOffset += nextElementOffset;
        }
    }
}

template <typename T>
void runMeanYdir(T* cudaImage, int offset, size_t x_num, size_t y_num, size_t z_num, cudaStream_t aStream, bool boundaryReflect) {
    dim3 threadsPerBlock(1, NumberOfWorkers, 1);
    dim3 numBlocks((x_num + threadsPerBlock.x - 1)/threadsPerBlock.x,
                   1,
                   (z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
    const int sharedMemorySize = sizeof(T) * y_num + (offset * 2 + 1) * sizeof(float);
    meanYdir<<<numBlocks,threadsPerBlock, sharedMemorySize, aStream>>>(cudaImage, offset, x_num, y_num, z_num, boundaryReflect);
}

template <typename T>
void runMeanXdir(T* cudaImage, int offset, size_t x_num, size_t y_num, size_t z_num, cudaStream_t aStream, bool boundaryReflect) {
    dim3 threadsPerBlock(1, NumberOfWorkers, 1);
    dim3 numBlocks(1,
                   (y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (z_num + threadsPerBlock.z - 1) / threadsPerBlock.z);
    // Shared memory size  - it is able to keep filter len elements for each worker.
    const int sharedMemorySize = (offset * 2 + 1) * sizeof(float) * NumberOfWorkers;
    meanXdir<<<numBlocks,threadsPerBlock, sharedMemorySize, aStream>>>(cudaImage, offset, x_num, y_num, z_num, boundaryReflect);
}

template <typename T>
void runMeanZdir(T* cudaImage, int offset, size_t x_num, size_t y_num, size_t z_num, cudaStream_t aStream, bool boundaryReflect) {
    dim3 threadsPerBlock(1, NumberOfWorkers, 1);
    dim3 numBlocks(1,
                   (y_num + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (x_num + threadsPerBlock.x - 1) / threadsPerBlock.x); // intentionally here for better memory readings
    // Shared memory size  - it is able to keep filter len elements for each worker.
    const int sharedMemorySize = (offset * 2 + 1) * sizeof(float) * NumberOfWorkers;
    meanZdir<<<numBlocks,threadsPerBlock, sharedMemorySize, aStream>>>(cudaImage, offset, x_num, y_num, z_num, boundaryReflect);
}

template <typename T>
void runMean(T *cudaImage, const PixelDataDim dim, int offsetX, int offsetY, int offsetZ, TypeOfMeanFlags flags, cudaStream_t aStream, bool boundaryReflect = false) {
    if (flags & MEAN_Y_DIR) {
        runMeanYdir(cudaImage, offsetY, dim.x, dim.y, dim.z, aStream, boundaryReflect);
    }

    if (flags & MEAN_X_DIR) {
        runMeanXdir(cudaImage, offsetX, dim.x, dim.y, dim.z, aStream, boundaryReflect);
    }

    if (flags & MEAN_Z_DIR) {
        runMeanZdir(cudaImage, offsetZ, dim.x, dim.y, dim.z, aStream, boundaryReflect);
    }
}

template <typename T>
__global__ void copy1D(const T *input, T *output, size_t len) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = input[idx];
    }
}

template <typename T>
void runCopy1D(const T *input, T *output, size_t len, cudaStream_t aStream) {
    dim3 threadsPerBlock(64);
    dim3 numBlocks((len + threadsPerBlock.x - 1) / threadsPerBlock.x);
    copy1D <<<numBlocks, threadsPerBlock, 0, aStream>>> (input, output, len);
}

template<typename T>
__global__ void absDiff1D(T *data, const T *reference, size_t len) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        data[idx] = abs(data[idx] - reference[idx]);
    }
}

template <typename T>
void runAbsDiff1D(T *data, const T *reference, size_t len, cudaStream_t aStream) {
    dim3 threadsPerBlock(64);
    dim3 numBlocks((len + threadsPerBlock.x - 1) / threadsPerBlock.x);
    absDiff1D <<< numBlocks, threadsPerBlock, 0, aStream >>> (data, reference, len);
}

template<typename T>
__global__ void rescale(T *data, size_t len, float varRescale) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float rescaled = varRescale * data[idx];
        data[idx] = rescaled;
    }
}

template <typename T>
void runRescale(T *data, size_t len, float varRescale, cudaStream_t aStream) {
    dim3 threadsPerBlock(64);
    dim3 numBlocks((len + threadsPerBlock.x - 1) / threadsPerBlock.x);
    rescale <<< numBlocks, threadsPerBlock, 0, aStream >>>(data, len, varRescale);
}

template <typename S>
__global__ void constantScale(S *image, size_t len) {
    // This is totally naive and slow implementation (only 1 thread is used) just to have CPU
    // code implemented in CUDA. This code will not be run in any normal usage of APR
    // and it is just here for sanity check and or super small images cases (like few pixels)
    // so DO NOT TRY TO OPTIMIZE IT - use your time for something more productive or have
    // some beers... still better than writing fast version of this code.

    float min_val = 660000;
    double sum = 0;

    for (size_t i = 0; i < len; ++i) {
        float tmp = image[i];

        sum += tmp;
        if (tmp < min_val) min_val = tmp;
    }

    float scale_val = (float) (sum / (float)len - min_val);

    for (size_t i = 0; i < len; ++i) {
        image[i] = scale_val;
    }
}

template <typename S>
void runConstantScale(S *image, PixelDataDim &dim) {
    // Check kernel description for further info!
    constantScale<<<1, 1>>>(image, dim.size());
}

template <typename T, typename S>
void runLocalIntensityScalePipeline(const PixelData<T> &image, const APRParameters &par, S *cudaImage, S *cudaTemp, cudaStream_t aStream) {
    float var_rescale;
    std::vector<int> var_win;
    auto lis = LocalIntensityScale();
    lis.get_window_alt(var_rescale, var_win, par, image);
    size_t win_y = var_win[0];
    size_t win_x = var_win[1];
    size_t win_z = var_win[2];
    size_t win_y2 = var_win[3];
    size_t win_x2 = var_win[4];
    size_t win_z2 = var_win[5];



    bool constant_scale = false;

    if (par.constant_intensity_scale || (lis.number_active_dimensions == 0)) {
        // include the case where the local intensity scale doesn't make sense due to the image being to small.
        // (This is for just edge cases and sanity checking)
        constant_scale = true;
    }

    PixelDataDim imageSize = image.getDimension();

    if (!constant_scale) {
        CudaMemoryUniquePtr<S> paddedImage;
        CudaMemoryUniquePtr<S> paddedTemp;
        PixelDataDim paddSize(std::max(win_y, win_y2), std::max(win_x, win_x2), std::max(win_z, win_z2));
        PixelDataDim paddedImageSize = imageSize + paddSize + paddSize; // padding on both ends of each dimension

        S *ci = cudaImage;
        S *ct = cudaTemp;
        PixelDataDim dim = image.getDimension();

        if (par.reflect_bc_lis) {
            // padd
            S *mem = nullptr;
            checkCuda(cudaMalloc(&mem, sizeof(S) * paddedImageSize.size()));
            paddedImage.reset(mem);
            mem = nullptr;
            checkCuda(cudaMalloc(&mem, sizeof(S) * paddedImageSize.size()));
            paddedTemp.reset(mem);

            runPaddPixels(cudaImage, paddedImage.get(), imageSize, paddedImageSize, paddSize, aStream);
            runPaddPixels(cudaTemp, paddedTemp.get(), imageSize, paddedImageSize, paddSize, aStream);

            ci = paddedImage.get();
            ct = paddedTemp.get();
            dim = paddedImageSize;
        }

        // Run LIS pipeline
        runCopy1D(ci, ct, dim.size(), aStream);
        runMean(ci, dim, win_x, win_y, win_z, MEAN_ALL_DIR, aStream, false);
        runAbsDiff1D(ci, ct, dim.size(), aStream);
        runMean(ci, dim, win_x2, win_y2, win_z2, MEAN_ALL_DIR, aStream, false);
        runRescale(ci, dim.size(), var_rescale, aStream);

        if (par.reflect_bc_lis) {
            // unpadd
            runUnpaddPixels(ci, cudaImage, paddedImageSize, imageSize, paddSize, aStream);
            runUnpaddPixels(ct, cudaTemp, paddedImageSize, imageSize, paddSize, aStream);
        }
    }
    else {
        runConstantScale(cudaImage, imageSize);
    }
}

template void runLocalIntensityScalePipeline<float,float>(const PixelData<float>&, const APRParameters&, float*, float*, cudaStream_t);



// =================================================== TEST helpers
// TODO: should be moved somewhere
template <typename T>
void calcMean(PixelData<T> &image, int offset, TypeOfMeanFlags flags, bool boundaryReflect) {
    cudaStream_t aStream = 0;

    ScopedCudaMemHandler<PixelData<T>, H2D | D2H> cudaImage(image, aStream);

    runMean(cudaImage.get(), image.getDimension(), offset, offset, offset, flags, 0, boundaryReflect);
}

// explicit instantiation of handled types
template void calcMean(PixelData<float>&, int, TypeOfMeanFlags, bool);
template void calcMean(PixelData<uint16_t>&, int, TypeOfMeanFlags, bool);


template <typename T>
void getLocalIntensityScale(PixelData<T> &image, PixelData<T> &temp, const APRParameters &par) {
    cudaStream_t aStream = 0;

    ScopedCudaMemHandler<PixelData<T>, H2D | D2H> cudaImage(image, aStream);
    ScopedCudaMemHandler<PixelData<T>, D2H> cudaTemp(temp, aStream);

    runLocalIntensityScalePipeline(image, par, cudaImage.get(), cudaTemp.get(), aStream);
}
template void getLocalIntensityScale(PixelData<float>&, PixelData<float>&, const APRParameters&);
