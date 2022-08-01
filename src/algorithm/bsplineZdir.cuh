#ifndef BSPLINE_Z_DIR_H
#define BSPLINE_Z_DIR_H


#include "cudaMisc.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>


/**
 * Runs bspline recursive filter in Z direction. Each processed 2D patch consist of number of workes
 * (distributed in Y direction) and each of them is handling the whole row in Z-dir.
 * Next patches are build on a top of first (first marked with letter P) until they process
 * whole y-dimension. Such a setup should be run for every plane in x-direction.
 *
 * Example block/threadblock calculation:
 *     constexpr int numOfWorkersY = 64;
 *     dim3 threadsPerBlock(1, numOfWorkersY, 1);
 *     dim3 numBlocks((input.x_num + threadsPerBlock.x - 1)/threadsPerBlock.x),
 *                    (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
 *                    1);
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
 *    n      X |\  X                            X
 *           X | \ X                            X
 *      z_num X|  \X                            X
 *         ^   X P X                            X
 *          \   X  X                            X
 *   z       \   X X                            X
 *   direction    X|                            X
 *                 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 *                 0                              x_num
 *                          X direction ->
 *
 * @tparam T - input image type
 * @param image - device pointer to image
 * @param x_num - dimension len in x-direction
 * @param y_num - dimension len in y-direction
 * @param z_num - dimension len in z-direction
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
__global__ void bsplineZdir(T *image, PixelDataDim dim,
                            const float *bc1, const float *bc2, const float *bc3, const float *bc4, size_t k0,
                            float b1, float b2, float norm_factor, bool *error) {

    const int yDirOffset = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t xDirOffset = (blockIdx.z * blockDim.z + threadIdx.z) * dim.y; // x is in 'z' to have good memory coalescing
    const size_t nextElementZdirOffset = dim.x * dim.y;
    const size_t dirLen = dim.z;
    const size_t minLen = min(dirLen, k0);

    if (yDirOffset < dim.y) {
        float temp1 = 0;
        float temp2 = 0;
        float temp3 = 0;
        float temp4 = 0;

        // calculate boundary values
        for (int k = 0; k < minLen; ++k) {
            T val = image[xDirOffset + k * nextElementZdirOffset + yDirOffset];
            temp1 += bc1[k] * val;
            temp2 += bc2[k] * val;
            val = image[xDirOffset + (dirLen - 1 - k) * nextElementZdirOffset + yDirOffset];
            temp3 += bc3[k] * val;
            temp4 += bc4[k] * val;
        }

        size_t errorCnt = 0;

        // set boundary values in two first and two last points processed direction
        image[xDirOffset + 0 * nextElementZdirOffset + yDirOffset] = round<T>(temp1, errorCnt);
        image[xDirOffset + 1 * nextElementZdirOffset + yDirOffset] = round<T>(temp2, errorCnt);
        image[xDirOffset + (dirLen - 2) * nextElementZdirOffset + yDirOffset] = round<T>(temp3 * norm_factor, errorCnt);
        image[xDirOffset + (dirLen - 1) * nextElementZdirOffset + yDirOffset] = round<T>(temp4 * norm_factor, errorCnt);

        // Causal Filter loop
        int64_t offset = xDirOffset + 2 * nextElementZdirOffset + yDirOffset;
        int64_t offsetLimit = xDirOffset + (dirLen - 2) * nextElementZdirOffset;
        while (offset < offsetLimit) {
            __syncthreads(); // only needed for speed imporovement (memory coalescing)
            const float temp = round<T>(image[offset] + b1 * temp2 + b2 * temp1, errorCnt);
            image[offset] = temp;
            temp1 = temp2;
            temp2 = temp;

            offset += nextElementZdirOffset;
        }

        // Anti-Causal Filter loop
        offset = xDirOffset + (dirLen - 3) * nextElementZdirOffset + yDirOffset;
        offsetLimit = xDirOffset;
        while (offset >= offsetLimit) {
            __syncthreads(); // only needed for speed imporovement (memory coalescing)
            const float temp = image[offset] + b1 * temp3 + b2 * temp4;
            image[offset] = round<T>(temp * norm_factor, errorCnt);
            temp4 = temp3;
            temp3 = temp;

            offset -= nextElementZdirOffset;
        }

        if (errorCnt > 0) *error = true;
    }
}

/**
 * Function for launching a kernel
 */
template<typename T>
void runBsplineZdir(T *cudaImage, PixelDataDim dim,
                    const float *bc1, const float *bc2, const float *bc3, const float *bc4,
                    size_t k0, float b1, float b2, float norm_factor, cudaStream_t aStream) {
    constexpr int numOfWorkersYdir = 128;
    dim3 threadsPerBlockZ(1, numOfWorkersYdir, 1);
    dim3 numBlocksZ(1,
                    (dim.y + threadsPerBlockZ.y - 1) / threadsPerBlockZ.y,
                    (dim.x + threadsPerBlockZ.x - 1) / threadsPerBlockZ.x);
    // In case of error this will be set to true by one of the kernels (CUDA does not guarantee which kernel will set global variable if more then one kernel
    // access it but this is enough for us to know that somewhere in one on more kernels overflow was detected.
    bool isErrorDetected = false;
    {
        ScopedCudaMemHandler<bool*, H2D | D2H> error(&isErrorDetected, 1);
        bsplineZdir<T> <<<numBlocksZ, threadsPerBlockZ, 0, aStream>>> (cudaImage, dim, bc1, bc2, bc3, bc4, k0, b1, b2, norm_factor, error.get());
    }

    if (isErrorDetected) {
        throw std::invalid_argument("integer under-/overflow encountered in CUDA bsplineZdir - "
                                    "try squashing the input image to a narrower range or use APRConverter<float>");
    }
}

#endif
