#ifndef BSPLINE_Y_DIR_H
#define BSPLINE_Y_DIR_H


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>
#include "cudaMisc.cuh"
#include "bsplineParams.h"


/**
 * Runs bspline recursive filter in Y direction - divided into two phases:
 * 1. calculate boundary conditions
 * 2. run recursive filter as a set of 2D patches:
 * Each processed 2D patch consist of number of workers
 * (distributed in Y direction) and each of them is handling the whole row in Y-dir.
 * Next patches are build on next to it in the x-dir to cover whole x * z domain.
 *
 *
 * Image memory setup is [z][x][y]
 *
 *     y_num
 *           XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 *           XX                            X
 *    y      X X                            X
 *           X  X                            X
 *    d      X   X                            X
 *    i      X    X                            X
 *    r ^    X     XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 *    e |    X     X                            X
 *    c |    X     X                            X
 *    t      X     X                            X
 *    i      X     X                            X
 *    o      X     X                            X
 *    n      X     X                            X
 *           X     X                            X
 *      z_num X    +------------+               X
 *         ^   X   |            |               X
 *          \   X  | ^          |               X
 *   z       \   X | | patch 0  |               X
 *   direction    X| |          |               X
 *                 +------------+XXXXXXXXXXXXXXXX
 *                 0                              x_num
 *                          X direction ->
 *
 * @tparam T - input image type
 * @param image - device pointer to image
 * @param x_num - dimension len in x-direction
 * @param y_num - dimension len in y-direction
 * @param bc1 - precomputed filter
 * @param bc2 - precomputed filter
 * @param bc3 - precomputed filter
 * @param bc4 - precomputed filter
 * @param k0 - filter len
 * @param b1 - filter coefficient
 * @param b2 - filter coefficient
 * @param norm_factor - filter norm factor
 */


template<typename T>
__global__ void bsplineYdirBoundary(T *image, PixelDataDim dim, BsplineParamsCuda p, float *boundary, bool *error) {
    const int xzIndexOfWorker = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int xzIndexOfBlock = (blockIdx.x * blockDim.x);

    const int numOfWorkers = blockDim.x;
    const int currentWorkerId = threadIdx.x;
    const size_t workersOffset = xzIndexOfBlock * dim.y; // per each (x,z) coordinate we have y-row

    const int64_t maxXZoffset = dim.x * dim.z;

    const size_t dirLen = dim.y;
    const size_t minLen = min(dirLen, p.k0);

    extern __shared__ float sharedMem[];
    float *bc1_vec2 = &sharedMem[0];
    float *bc2_vec2 = &bc1_vec2[p.k0];
    float *cache = (float*)&bc2_vec2[p.k0];

    // Read from global mem to cache
    for (int i = currentWorkerId; i < p.k0 * numOfWorkers; i += numOfWorkers) {
        if (i < p.k0) {
            bc1_vec2[i] = p.bc1[i];
            bc2_vec2[i] = p.bc2[i];
        }
        int offs = i % p.k0;
        int work = i / p.k0;
        if (work + xzIndexOfBlock < maxXZoffset) {
            cache[work * p.k0 + offs] = image[workersOffset + dim.y * work + offs];
        }
    }
    __syncthreads();

    //forwards direction
    if (xzIndexOfWorker < dim.x * dim.z) {
        float temp1 = 0;
        float temp2 = 0;
        for (size_t k = 0; k < minLen; ++k) {
            temp1 += bc1_vec2[k] * (T)cache[currentWorkerId * p.k0 + k];
            temp2 += bc2_vec2[k] * (T)cache[currentWorkerId * p.k0 + k];
        }
        boundary[xzIndexOfWorker*4 + 0] = temp1;
        boundary[xzIndexOfWorker*4 + 1] = temp2;
    }

    // ----------------- second end
    __syncthreads();

    for (int i = currentWorkerId; i < p.k0 * numOfWorkers; i += numOfWorkers) {
        if (i < p.k0) {
            bc1_vec2[i] = p.bc3[i];
            bc2_vec2[i] = p.bc4[i];
        }
        int offs = i % p.k0;
        int work = i / p.k0;
        if (work + xzIndexOfBlock < maxXZoffset) {
            cache[work * p.k0 + offs] = image[workersOffset + dim.y * work + dim.y - 1 - offs];
        }
    }
    __syncthreads();

    size_t errorCnt = 0;

    //forwards direction
    if (xzIndexOfWorker < dim.x * dim.z) {
        float temp3 = 0;
        float temp4 = 0;
        for (size_t k = 0; k < minLen; ++k) {
            temp3 += bc1_vec2[k] * (T)cache[currentWorkerId * p.k0 + k];
            temp4 += bc2_vec2[k] * (T)cache[currentWorkerId * p.k0 + k];
        }
        boundary[xzIndexOfWorker*4 + 2] = round<T>(temp3 * p.norm_factor, errorCnt);
        boundary[xzIndexOfWorker*4 + 3] = round<T>(temp4 * p.norm_factor, errorCnt);
    }

    if (errorCnt > 0) *error = true;
}

constexpr int blockWidth = 32;
constexpr int numOfThreads = 32;
extern __shared__ char sharedMemProcess[];
template<typename T>
__global__ void bsplineYdirProcess(T *image, const PixelDataDim dim, BsplineParamsCuda p, float *boundary, bool *error) {
    const int numOfWorkers = blockDim.x;
    const int currentWorkerId = threadIdx.x;
    const int xzOffset = blockIdx.x * blockDim.x;
    const int64_t maxXZoffset = dim.x * dim.z;
    const int64_t workersOffset = xzOffset * dim.y;

    float (*cache)[blockWidth + 0] = (float (*)[blockWidth + 0]) &sharedMemProcess[0];

    float temp1, temp2;
    size_t errorCnt = 0;

    // ---------------- forward direction -------------------------------------------
    for (int yBlockBegin = 0; yBlockBegin < dim.y - 2; yBlockBegin += blockWidth) {

        // Read from global mem to cache
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (offs + yBlockBegin < (dim.y - 2) && work + xzOffset < maxXZoffset) {
                cache[work][(offs + work)%blockWidth] = image[workersOffset + dim.y * work + offs + yBlockBegin];
            }
        }
        __syncthreads();

        // Do operations
        if (xzOffset + currentWorkerId < maxXZoffset) {
            if (yBlockBegin == 0) {
                temp1 = boundary[(xzOffset + currentWorkerId) * 4 + 0];
                temp2 = boundary[(xzOffset + currentWorkerId) * 4 + 1];
                cache[currentWorkerId][(0 + currentWorkerId)%blockWidth] = temp1;
                cache[currentWorkerId][(1 + currentWorkerId)%blockWidth] = temp2;
            }
            for (size_t k = yBlockBegin == 0 ? 2 : 0; k < blockWidth && k + yBlockBegin < dim.y - 2; ++k) {
                float  temp = temp2*p.b1 + temp1*p.b2 + (T)cache[currentWorkerId][(k + currentWorkerId)%blockWidth];
                cache[currentWorkerId][(k + currentWorkerId)%blockWidth] = temp;
                temp1 = temp2;
                temp2 = temp;
            }
        }
        __syncthreads();

        // Write from cache to global mem
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (offs + yBlockBegin < (dim.y - 2) && work + xzOffset < maxXZoffset) {
                image[workersOffset + dim.y * work + offs + yBlockBegin] = round<T>(cache[work][(offs + work)%blockWidth], errorCnt);
            }
        }
        __syncthreads();
    }

    // ---------------- backward direction -------------------------------------------
    for (int yBlockBegin = dim.y - 1; yBlockBegin >= 0; yBlockBegin -= blockWidth) {

        // Read from global mem to cache
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (yBlockBegin - offs >= 0 && work + xzOffset < maxXZoffset) {
                cache[work][(offs + work)%blockWidth] = image[workersOffset + dim.y * work - offs + yBlockBegin];
            }
        }
        __syncthreads();

        // Do operations
        if (xzOffset + currentWorkerId < maxXZoffset) {
            if (yBlockBegin == dim.y - 1) {
                temp1 = boundary[(xzOffset + currentWorkerId) * 4 + 3] / p.norm_factor;
                temp2 = boundary[(xzOffset + currentWorkerId) * 4 + 2] / p.norm_factor;
                cache[currentWorkerId][(0 + currentWorkerId)%blockWidth] = p.norm_factor * temp1;
                cache[currentWorkerId][(1 + currentWorkerId)%blockWidth] = p.norm_factor * temp2;
            }
            for (int64_t k = yBlockBegin == dim.y - 1 ? 2 : 0; k < blockWidth && yBlockBegin - k >= 0; ++k) {
                float  temp = temp2*p.b1 + temp1*p.b2 + (T)cache[currentWorkerId][(k + currentWorkerId)%blockWidth];
                cache[currentWorkerId][(k + currentWorkerId)%blockWidth] = temp * p.norm_factor;
                temp1 = temp2;
                temp2 = temp;
            }
        }
        __syncthreads();

        // Write from cache to global mem
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (yBlockBegin - offs >= 0 && work + xzOffset < maxXZoffset) {
                image[workersOffset + dim.y * work - offs + yBlockBegin] = round<T>(cache[work][(offs + work)%blockWidth], errorCnt);
            }
        }
        __syncthreads();
    }

    if (errorCnt > 0) *error = true;
}

/**
 * Function for launching a kernel
 */
template <typename T>
void runBsplineYdir(T *cudaImage, PixelDataDim dim, BsplineParamsCuda &p, float *boundary, cudaStream_t aStream) {

    dim3 threadsPerBlock(numOfThreads);
    dim3 numBlocks((dim.x * dim.z + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemSize = (2 /*bc vectors*/) * (p.k0) * sizeof(float) + numOfThreads * (p.k0) * sizeof(float);
    bool isErrorDetected = false;
    {
        ScopedCudaMemHandler<bool *, H2D | D2H> error(&isErrorDetected, 1);
        bsplineYdirBoundary<T> <<< numBlocks, threadsPerBlock, sharedMemSize, aStream >>>(cudaImage, dim, p, boundary, error.get());
        sharedMemSize = numOfThreads * blockWidth * sizeof(float);
        bsplineYdirProcess<T> <<< numBlocks, threadsPerBlock, sharedMemSize, aStream >>>(cudaImage, dim, p, boundary, error.get());
    }

    if (isErrorDetected) {
        throw std::invalid_argument("integer under-/overflow encountered in CUDA bsplineYdir - "
                                    "try squashing the input image to a narrower range or use APRConverter<float>");
    }
}
#endif
