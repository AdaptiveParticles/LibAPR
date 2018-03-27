#ifndef BSPLINE_Y_DIR_H
#define BSPLINE_Y_DIR_H


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>

/**
 * Runs bspline recursive filter in Y direction. Each processed 2D patch consist of number of workes (distributed in Y direction)
 * and each of them is handling the whole row in X-dir.
 * Next patches are build on a top of first (like patch1 in example below) and they cover
 * whole y-dimension. Such a setup should be run for every plane in z-direction.
 *
 * Example block/threadblock calculation:
 *     constexpr int numOfWorkersY = 64;
 *     dim3 threadsPerBlock(1, numOfWorkersY, 1);
 *     dim3 numBlocks(1,
 *                    (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
 *                    (input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
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
 *    c |    X     X  ...                       X
 *    t      X     +----------------------------+                   X
 *    i      X     |                            |
 *    o      X     | ----->                     |
 *    n      X     | patch 1                    |
 *           X     |                            |
 *      z_num X    +----------------------------+
 *         ^   X   |                            |
 *          \   X  | ----->                     |
 *   z       \   X | patch 0                    |
 *   direction    X|                            |
 *                 +----------------------------+
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


extern __shared__ float sharedMem[];
template<typename T>
__global__ void bsplineYdirBoundary(T *image, size_t x_num, size_t y_num, size_t z_num,
                                    const float *bc1_vec, const float *bc2_vec, const float *bc3_vec, const float *bc4_vec,
                                    size_t k0, float *boundary) {
    const int xzIndexOfWorker = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int xzIndexOfBlock = (blockIdx.x * blockDim.x);

    const int numOfWorkers = blockDim.x;
    const int currentWorkerId = threadIdx.x;
    const size_t workersOffset = xzIndexOfBlock * y_num; // per each (x,z) coordinate we have y-row

    const int64_t maxXZoffset = x_num * z_num;

    float *bc1_vec2 = &sharedMem[0];
    float *bc2_vec2 = &bc1_vec2[k0];
    T *cache = (T*)&bc2_vec2[k0];

    // Read from global mem to cache
    for (int i = currentWorkerId; i < k0 * numOfWorkers; i += numOfWorkers) {
        if (i < k0) {
            bc1_vec2[i] = bc1_vec[i];
            bc2_vec2[i] = bc2_vec[i];
        }
        int offs = i % k0;
        int work = i / k0;
        if (work + xzIndexOfBlock < maxXZoffset) {
            cache[work * k0 + offs] = image[workersOffset + y_num * work + offs];
        }
    }
    __syncthreads();

    //forwards direction
    if (xzIndexOfWorker < x_num * z_num) {
        float temp1 = 0;
        float temp2 = 0;
        for (size_t k = 0; k < k0; ++k) {
            temp1 += bc1_vec2[k] * cache[currentWorkerId * k0 + k];
            temp2 += bc2_vec2[k] * cache[currentWorkerId * k0 + k];
        }
        boundary[xzIndexOfWorker*4 + 0] = temp2;
        boundary[xzIndexOfWorker*4 + 1] = temp1;
    }

    // ----------------- second end
    __syncthreads();

    for (int i = currentWorkerId; i < k0 * numOfWorkers; i += numOfWorkers) {
        if (i < k0) {
            bc1_vec2[i] = bc3_vec[i];
            bc2_vec2[i] = bc4_vec[i];
        }
        int offs = i % k0;
        int work = i / k0;
        if (work + xzIndexOfBlock < maxXZoffset) {
            cache[work * k0 + offs] = image[workersOffset + y_num * work + y_num - 1 - offs];
        }
    }
    __syncthreads();

    //forwards direction
    if (xzIndexOfWorker < x_num * z_num) {
        float temp3 = 0;
        float temp4 = 0;
        for (size_t k = 0; k < k0; ++k) {
            temp3 += bc1_vec2[k] * cache[currentWorkerId * k0 + k];
            temp4 += bc2_vec2[k] * cache[currentWorkerId * k0 + k];
        }
        boundary[xzIndexOfWorker*4 + 2] = temp3;
        boundary[xzIndexOfWorker*4 + 3] = temp4;
    }
}

constexpr int blockWidth = 32;
constexpr int numOfThreads = 32;
extern __shared__ char sharedMemProcess[];
template<typename T>
__global__ void bsplineYdirProcess(T *image, const size_t x_num, const size_t y_num, const size_t z_num, size_t k0,
                                   const float b1, const float b2, const float norm_factor, float *boundary) {
    const int numOfWorkers = blockDim.x;
    const int currentWorkerId = threadIdx.x;
    const int xzOffset = blockIdx.x * blockDim.x;
    const int64_t maxXZoffset = x_num * z_num;
    const int64_t workersOffset = xzOffset * y_num;

    T (*cache)[blockWidth + 0] = (T (*)[blockWidth + 0]) &sharedMemProcess[0];

    float temp1, temp2;

    // ---------------- forward direction -------------------------------------------
    for (int yBlockBegin = 0; yBlockBegin < y_num - 2; yBlockBegin += blockWidth) {

        // Read from global mem to cache
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (offs + yBlockBegin < (y_num - 2) && work + xzOffset < maxXZoffset) {
                cache[work][(offs + work)%blockWidth] = image[workersOffset + y_num * work + offs + yBlockBegin];
            }
        }
        __syncthreads();

        // Do operations
        if (xzOffset + currentWorkerId < maxXZoffset) {
            if (yBlockBegin == 0) {
                temp2 = boundary[(xzOffset + currentWorkerId) * 4 + 0];
                temp1 = boundary[(xzOffset + currentWorkerId) * 4 + 1];
                cache[currentWorkerId][(0 + currentWorkerId)%blockWidth] = temp2;
                cache[currentWorkerId][(1 + currentWorkerId)%blockWidth] = temp1;
            }
            for (size_t k = yBlockBegin == 0 ? 2 : 0; k < blockWidth && k + yBlockBegin < y_num - 2; ++k) {
                float  temp = temp1*b1 + temp2*b2 + cache[currentWorkerId][(k + currentWorkerId)%blockWidth];
                cache[currentWorkerId][(k + currentWorkerId)%blockWidth] = temp;
                temp2 = temp1;
                temp1 = temp;
            }
        }
        __syncthreads();

        // Write from cache to global mem
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (offs + yBlockBegin < (y_num - 2) && work + xzOffset < maxXZoffset) {
                image[workersOffset + y_num * work + offs + yBlockBegin] = cache[work][(offs + work)%blockWidth];
            }
        }
        __syncthreads();
    }

    // ---------------- backward direction -------------------------------------------
    for (int yBlockBegin = y_num - 1; yBlockBegin >= 0; yBlockBegin -= blockWidth) {

        // Read from global mem to cache
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (yBlockBegin - offs >= 0 && work + xzOffset < maxXZoffset) {
                cache[work][(offs + work)%blockWidth] = image[workersOffset + y_num * work - offs + yBlockBegin];
            }
        }
        __syncthreads();

        // Do operations
        if (xzOffset + currentWorkerId < maxXZoffset) {
            if (yBlockBegin == y_num - 1) {
                temp2 = boundary[(xzOffset + currentWorkerId) * 4 + 3];
                temp1 = boundary[(xzOffset + currentWorkerId) * 4 + 2];
                cache[currentWorkerId][(0 + currentWorkerId)%blockWidth] = norm_factor * temp2;
                cache[currentWorkerId][(1 + currentWorkerId)%blockWidth] = norm_factor * temp1;
            }
            for (int64_t k = yBlockBegin == y_num - 1 ? 2 : 0; k < blockWidth && yBlockBegin - k >= 0; ++k) {
                float  temp = temp1*b1 + temp2*b2 + cache[currentWorkerId][(k + currentWorkerId)%blockWidth];
                cache[currentWorkerId][(k + currentWorkerId)%blockWidth] = temp * norm_factor;
                temp2 = temp1;
                temp1 = temp;
            }
        }
        __syncthreads();

        // Write from cache to global mem
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (yBlockBegin - offs >= 0 && work + xzOffset < maxXZoffset) {
                image[workersOffset + y_num * work - offs + yBlockBegin] = cache[work][(offs + work)%blockWidth];
            }
        }
        __syncthreads();
    }
}

#endif
