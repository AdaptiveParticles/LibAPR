#ifndef LIBAPR_PADDPIXELDATA_CUH
#define LIBAPR_PADDPIXELDATA_CUH


#include "data_structures/Mesh/PixelData.hpp"


template <typename T>
__global__ void pad(const T* input, T *output, size_t inY, size_t inX, size_t inZ, size_t padY, size_t padX, size_t padZ) {
    size_t yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t zIdx = blockIdx.z * blockDim.z + threadIdx.z;

    // input cube
    size_t inputIdx = (zIdx * inX + xIdx) * inY + yIdx;

    // output cube (shifted by pad size from all sides)
    size_t outputIdx = ((zIdx + padZ) * (inX + 2 * padX) + (xIdx + padX)) * (inY + 2 * padY) + (yIdx + padY);

    // copy input data to interior
    if (yIdx < inY && xIdx < inX && zIdx < inZ) {
        output[outputIdx] = input[inputIdx];
    }
}

template <typename T>
void runPad(const T* input, T *output, const PixelDataDim &dataSize, const PixelDataDim &padSize, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 64, 1);
    dim3 numBlocks((dataSize.x() + threadsPerBlock.x - 1)/threadsPerBlock.x,
                   (dataSize.y() + threadsPerBlock.y - 1)/threadsPerBlock.y,
                   (dataSize.z() + threadsPerBlock.z - 1)/threadsPerBlock.z);
    pad<<<numBlocks,threadsPerBlock, 0, aStream>>>(input, output, dataSize.y(), dataSize.x(), dataSize.z(), padSize.y(), padSize.x(), padSize.z());
}


#endif
