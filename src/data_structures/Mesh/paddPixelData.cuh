#ifndef LIBAPR_PADDPIXELDATA_CUH
#define LIBAPR_PADDPIXELDATA_CUH


#include "data_structures/Mesh/PixelData.hpp"


template <typename T>
__global__ void paddPixels(const T* input, T *output, const PixelDataDim inputSize, const PixelDataDim outputSize, const PixelDataDim padSize) {
    size_t yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t zIdx = blockIdx.z * blockDim.z + threadIdx.z;

    // copy data to output (padded) cube
    if (yIdx < outputSize.y && xIdx < outputSize.x && zIdx < outputSize.z) {

        // output cube index
        size_t outputIdx = (zIdx * outputSize.x + xIdx) * outputSize.y + yIdx;

        // input cube index
        int yIn = yIdx - padSize.y;
        if (yIn < 0) yIn = -yIn;                                      // reflected boundary on LHS
        if (yIn >= inputSize.y) yIn -= 2 * (yIn - (inputSize.y - 1)); // reflected boundary on RHS

        int xIn = xIdx - padSize.x;
        if (xIn < 0) xIn = -xIn;                                      // reflected boundary on LHS
        if (xIn >= inputSize.x) xIn -= 2 * (xIn - (inputSize.x - 1)); // reflected boundary on RHS

        int zIn = zIdx - padSize.z;
        if (zIn < 0) zIn = -zIn;                                      // reflected boundary on LHS
        if (zIn >= inputSize.z) zIn -= 2 * (zIn - (inputSize.z - 1)); // reflected boundary on RHS

        size_t inputIdx = (zIn * inputSize.x + xIn) * inputSize.y + yIn;

        output[outputIdx] = input[inputIdx];
    }
}

template <typename T>
void runPaddPixels(const T* input, T *output, const PixelDataDim &inputSize, const PixelDataDim &outputSize, const PixelDataDim &padSize, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 64, 1);
    dim3 numBlocks((outputSize.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (outputSize.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (outputSize.z + threadsPerBlock.z - 1) / threadsPerBlock.z);

    paddPixels<<<numBlocks, threadsPerBlock, 0, aStream>>>(input, output, inputSize, outputSize, padSize);
}

template <typename T>
__global__ void unpaddPixels(const T* input, T *output, const PixelDataDim inputSize, const PixelDataDim outputSize, const PixelDataDim padSize) {
    size_t yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t zIdx = blockIdx.z * blockDim.z + threadIdx.z;

    // copy data to output (unpadded) cube
    if (yIdx < outputSize.y && xIdx < outputSize.x && zIdx < outputSize.z) {

        // output cube index
        size_t outputIdx = (zIdx * outputSize.x + xIdx) * outputSize.y + yIdx;

        // input cube index (map coordinates of output cube to internal cube of padded cube)
        int yIn = yIdx + padSize.y;
        int xIn = xIdx + padSize.x;
        int zIn = zIdx + padSize.z;
        size_t inputIdx = (zIn * inputSize.x + xIn) * inputSize.y + yIn;

        output[outputIdx] = input[inputIdx];
    }
}

template <typename T>
void runUnpaddPixels(const T* input, T *output, const PixelDataDim &inputSize, const PixelDataDim &outputSize, const PixelDataDim &padSize, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 64, 1);
    dim3 numBlocks((outputSize.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (outputSize.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (outputSize.z + threadsPerBlock.z - 1) / threadsPerBlock.z);

    unpaddPixels<<<numBlocks, threadsPerBlock, 0, aStream>>>(input, output, inputSize, outputSize, padSize);
}

#endif
