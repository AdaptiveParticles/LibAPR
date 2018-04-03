#ifndef BSPLINE_X_DIR_H
#define BSPLINE_X_DIR_H


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>

/**
 * Runs bspline recursive filter in X direction. Each processed 2D patch consist of number of workes (distributed in Y direction)
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
template<typename T>
__global__ void bsplineXdir(T *image, size_t x_num, size_t y_num,
                            const float *bc1, const float *bc2, const float *bc3, const float *bc4, size_t k0,
                            float b1, float b2, float norm_factor) {

    const int yDirOffset = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zDirOffset = (blockIdx.z * blockDim.z + threadIdx.z) * x_num * y_num;
    const size_t nextElementXdirOffset = y_num;
    const size_t dirLen = x_num;

    if (yDirOffset < y_num) {
        float temp1 = 0;
        float temp2 = 0;
        float temp3 = 0;
        float temp4 = 0;
        // calculate boundary values
        for (int k = 0; k < k0; ++k) {
            T val = image[zDirOffset + k * nextElementXdirOffset + yDirOffset];
            temp1 += bc1[k] * val;
            temp2 += bc2[k] * val;
            val = image[zDirOffset + (dirLen - 1 - k) * nextElementXdirOffset + yDirOffset];
            temp3 += bc3[k] * val;
            temp4 += bc4[k] * val;
        }

        // set boundary values in two first and two last points processed direction
        image[zDirOffset + 0 * nextElementXdirOffset + yDirOffset] = temp1;
        image[zDirOffset + 1 * nextElementXdirOffset + yDirOffset] = temp2;
        image[zDirOffset + (dirLen - 2) * nextElementXdirOffset + yDirOffset] = temp3 * norm_factor;
        image[zDirOffset + (dirLen - 1) * nextElementXdirOffset + yDirOffset] = temp4 * norm_factor;

        // Causal Filter loop
        int64_t offset = zDirOffset + 2 * nextElementXdirOffset + yDirOffset;
        int64_t offsetLimit = zDirOffset + (dirLen - 2) * nextElementXdirOffset;
        do {
            const float temp = temp1 * b2 + temp2 * b1 + image[offset];
            image[offset] = temp;
            temp1 = temp2;
            temp2 = temp;

            offset += nextElementXdirOffset;
        } while (offset < offsetLimit);

        // Anti-Causal Filter loop
        offset = zDirOffset + (dirLen - 3) * nextElementXdirOffset + yDirOffset;
        offsetLimit = zDirOffset;
        do {
            // do calculations and store
            const float temp = temp3 * b1 + temp4 * b2 + image[offset];
            image[offset] = temp * norm_factor;
            temp4 = temp3;
            temp3 = temp;

            offset -= nextElementXdirOffset;
        } while (offset >= offsetLimit);
    }
}


#endif
