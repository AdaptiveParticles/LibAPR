#ifndef BSPLINE_X_DIR_H
#define BSPLINE_X_DIR_H


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>
#include "cudaMisc.cuh"
#include "bsplineParams.h"

/**
 * Runs bspline recursive filter in X direction. Each processed 2D patch consist of number of workers
 * (distributed in Y direction) and each of them is handling the whole row in X-dir.
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
template<typename T>
__global__ void bsplineXdir(T *image, PixelDataDim dim, BsplineParamsCuda p, bool *error) {

    const int yDirOffset = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zDirOffset = (blockIdx.z * blockDim.z + threadIdx.z) * dim.x * dim.y;
    const size_t nextElementXdirOffset = dim.y;
    const size_t dirLen = dim.x;
    const size_t minLen = min(dirLen, p.k0);

    if (yDirOffset < dim.y) {
        float temp1 = 0;
        float temp2 = 0;
        float temp3 = 0;
        float temp4 = 0;

        // calculate boundary values
        for (int k = 0; k < minLen; ++k) {
            T val = image[zDirOffset + k * nextElementXdirOffset + yDirOffset];
            temp1 += p.bc1[k] * val;
            temp2 += p.bc2[k] * val;
            val = image[zDirOffset + (dirLen - 1 - k) * nextElementXdirOffset + yDirOffset];
            temp3 += p.bc3[k] * val;
            temp4 += p.bc4[k] * val;
        }

        size_t errorCnt = 0;

        // set boundary values in two first and two last points processed direction
        image[zDirOffset + 0 * nextElementXdirOffset + yDirOffset] = round<T>(temp1, errorCnt);
        image[zDirOffset + 1 * nextElementXdirOffset + yDirOffset] = round<T>(temp2, errorCnt);
        image[zDirOffset + (dirLen - 2) * nextElementXdirOffset + yDirOffset] = round<T>(temp3 * p.norm_factor, errorCnt);
        image[zDirOffset + (dirLen - 1) * nextElementXdirOffset + yDirOffset] = round<T>(temp4 * p.norm_factor, errorCnt);

        // Causal Filter loop
        int64_t offset = zDirOffset + 2 * nextElementXdirOffset + yDirOffset;
        int64_t offsetLimit = zDirOffset + (dirLen - 2) * nextElementXdirOffset;
        while (offset < offsetLimit) {
            __syncthreads(); // only needed for speed imporovement (memory coalescing)
            const float temp = round<T>(image[offset] + p.b1 * temp2 + p.b2 * temp1, errorCnt);
            image[offset] = temp;
            temp1 = temp2;
            temp2 = temp;

            offset += nextElementXdirOffset;
        }

        // Anti-Causal Filter loop
        offset = zDirOffset + (dirLen - 3) * nextElementXdirOffset + yDirOffset;
        offsetLimit = zDirOffset;
        while (offset >= offsetLimit) {
            __syncthreads(); // only needed for speed imporovement (memory coalescing)
            const float temp = image[offset] + p.b1 * temp3 + p.b2 * temp4;
            image[offset] = round<T>(temp * p.norm_factor, errorCnt);
            temp4 = temp3;
            temp3 = temp;

            offset -= nextElementXdirOffset;
        }

        if (errorCnt > 0) *error = true;
    }
}

/**
 * Function for launching a kernel
 */
template<typename T>
void runBsplineXdir(T *cudaImage, PixelDataDim dim, BsplineParamsCuda &p, cudaStream_t aStream) {
    constexpr int numOfWorkersYdir = 128;
    dim3 threadsPerBlockX(1, numOfWorkersYdir, 1);
    dim3 numBlocksX(1,
                    (dim.y + threadsPerBlockX.y - 1) / threadsPerBlockX.y,
                    (dim.z + threadsPerBlockX.z - 1) / threadsPerBlockX.z);
    // In case of error this will be set to true by one of the kernels (CUDA does not guarantee which kernel will set global variable if more then one kernel
    // access it but this is enough for us to know that somewhere in one on more kernels overflow was detected.
    bool isErrorDetected = false;
    {
        ScopedCudaMemHandler<bool*, H2D | D2H> error(&isErrorDetected, 1);
        bsplineXdir<T> <<<numBlocksX, threadsPerBlockX, 0, aStream>>>(cudaImage, dim, p, error.get());
    }

    if (isErrorDetected) {
        throw std::invalid_argument("integer under-/overflow encountered in CUDA bsplineXdir - "
                                    "try squashing the input image to a narrower range or use APRConverter<float>");
    }
}

#endif
