#include "PullingSchemeCuda.hpp"

#include <cuda_runtime.h>

#include "misc/CudaTools.cuh"
#include "data_structures/Mesh/downsample.cuh"
#include "algorithm/OVPC.h"


template <typename T, typename S>
__global__ void copyAndClampLevels(const T *input, S *output, size_t length, int levelMin, int levelMax) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < length) {
        T v = input[idx];
        if (v > levelMax) v = levelMax;
        if (v < levelMin) v = levelMin;
        output[idx] = v;
    }
}

template <typename T, typename S>
void runCopyAndClampLevels(T *inputData, S *outputData, size_t lenght, int levelMin, int levelMax, cudaStream_t aStream) {
    dim3 threadsPerBlock(128);
    dim3 numBlocks((lenght + threadsPerBlock.x - 1)/threadsPerBlock.x);
    copyAndClampLevels<<<numBlocks,threadsPerBlock, 0, aStream>>>(inputData, outputData, lenght, levelMin, levelMax);
};


template <typename T>
__global__ void oneLevel(T *data, size_t xLen, size_t yLen, size_t zLen, int level) {
    const int xi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int yi = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int zi = (blockIdx.z * blockDim.z) + threadIdx.z;
    if (xi >= xLen || yi >= yLen || zi >= zLen) return;

    int xmin = xi > 0 ? xi - 1 : 0;
    int xmax = xi < xLen - 1 ? xi + 1 : xLen - 1;
    int ymin = yi > 0 ? yi - 1 : 0;
    int ymax = yi < yLen - 1 ? yi + 1 : yLen - 1;
    int zmin = zi > 0 ? zi - 1 : 0;
    int zmax = zi < zLen - 1 ? zi + 1 : zLen - 1;

    bool ok = true;
    bool neig = false;
    for (int z = zmin; z <= zmax; ++z) {
        for (int x = xmin; x <= xmax; ++x) {
            for (int y = ymin; y <= ymax; ++y) {
                const size_t idx = z * xLen * yLen + x * yLen + y;
                T currentLevel = ~OVPC::MASK & data[idx];
                if (currentLevel > level) { ok = false; break; }
                else if (currentLevel == level) neig = true;
            }
        }
    }
    if (ok) {
        const size_t idx = zi * xLen * yLen + xi * yLen + yi;
        T status = data[idx];
        if (status == level) data[idx] |= OVPC::SEED;
        else if (neig) data[idx] |= OVPC::BOUNDARY;
        else data[idx] |= OVPC::FILLER;
    }
}

template <typename T>
void runOneLevel(T *data, size_t xLen, size_t yLen, size_t zLen, int level, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 128, 1);
    dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (yLen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (zLen + threadsPerBlock.z - 1) / threadsPerBlock.z);
//    dim3 numBlocks((xLen * yLen * zLen + threadsPerBlock.x - 1)/threadsPerBlock.x);
    oneLevel<<<numBlocks,threadsPerBlock, 0, aStream>>>(data, xLen, yLen, zLen, level);
};

template <typename T>
__global__ void secondPhase(T *data, T *child, size_t xLen, size_t yLen, size_t zLen, size_t xLenc, size_t yLenc, size_t zLenc, bool isLevelMax) {
    const int xi = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int yi = (blockIdx.y * blockDim.y) + threadIdx.y;
    const int zi = (blockIdx.z * blockDim.z) + threadIdx.z;
    if (xi >= xLen || yi >= yLen || zi >= zLen) return;

    int xmin = 2 * xi;
    int xmax = 2 * xi + 1; xmax = xmax >= xLenc ? xLenc - 1 : xmax;
    int ymin = 2 * yi;
    int ymax = 2 * yi + 1; ymax = ymax >= yLenc ? yLenc - 1 : ymax;
    int zmin = 2 * zi;
    int zmax = 2 * zi + 1; zmax = zmax >= zLenc ? zLenc - 1 : zmax;


    uint8_t status = data[zi * xLen * yLen + xi * yLen + yi];

    for (int z = zmin; z <= zmax; ++z) {
        for (int x = xmin; x <= xmax; ++x) {
            for (int y = ymin; y <= ymax; ++y) {
                size_t children_index = z * xLenc * yLenc + x * yLenc + y;
                child[children_index] = status >= (OVPC::OVPC_SEED << OVPC::BIT_SHIFT) ? 0 : child[children_index] >> OVPC::BIT_SHIFT;
            }
        }
    }
    if (isLevelMax) data[zi * xLen * yLen + xi * yLen + yi] = status >> OVPC::BIT_SHIFT;
}

template <typename T>
void runSecondPhase(T *data, T *child, size_t xLen, size_t yLen, size_t zLen, size_t xLenc, size_t yLenc, size_t zLenc, bool isLevelMax, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 128, 1);
    dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (yLen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (zLen + threadsPerBlock.z - 1) / threadsPerBlock.z);
    secondPhase<<<numBlocks,threadsPerBlock, 0, aStream>>>(data, child, xLen, yLen, zLen, xLenc, yLenc, zLenc, isLevelMax);
};

// explicit instantiation of handled types
template void computeOVPC(const PixelData<float>&, PixelData<TreeElementType>&, int, int);

template <typename T, typename S>
void computeOVPC(const PixelData<T> &input, PixelData<S> &output, int levelMin, int levelMax) {


    ScopedCudaMemHandler<const PixelData<T>, H2D> in(input);
    ScopedCudaMemHandler<PixelData<S>, D2H> mem(output);


    CudaTimer t(true, "OVPCCUDA");

    t.start_timer("wait");
    waitForCuda();
    t.stop_timer();

    t.start_timer("ALL");
    // TODO: This is not needed later - just for having clear debug
    //cudaMemset(mem.get(), 0, mem.getNumOfBytes());

    // =============== Create pyramid
    std::vector<S*> levels(levelMax + 1, nullptr);
    std::vector<size_t> xSize(levelMax + 1);
    std::vector<size_t> ySize(levelMax + 1);
    std::vector<size_t> zSize(levelMax + 1);

    int xDS = input.x_num;
    int yDS = input.y_num;
    int zDS = input.z_num;

    size_t offset = 0;
    for (int l = levelMax; l >= levelMin; --l) {
        levels[l] = reinterpret_cast<TreeElementType *>(mem.get()) + offset;
        xSize[l] = xDS;
        ySize[l] = yDS;
        zSize[l] = zDS;

        offset += xDS * yDS * zDS * sizeof(TreeElementType);
        // round up to 16-bytes
        const size_t alignemet = 16;
        offset = ((offset + alignemet - 1) / alignemet ) * alignemet;

        xDS = ceil(xDS/2.0);
        yDS = ceil(yDS/2.0);
        zDS = ceil(zDS/2.0);
    }

    runCopyAndClampLevels(in.get(), levels[levelMax], in.getSize(), levelMin, levelMax, 0);

    for (int l = levelMax - 1; l >= levelMin; --l) {
        runDownsampleMax(levels[l + 1], levels[l], xSize[l + 1], ySize[l + 1], zSize[l + 1], 0);
    }


    // ================== Phase 1 - top to down
    for (int l = levelMin; l <= levelMax; ++l) {
        runOneLevel(levels[l], xSize[l], ySize[l], zSize[l], l, 0);
    }
    // ================== Phase 1 - down to top
    for (int l = levelMax - 1; l >= levelMin; --l) {
        runSecondPhase(levels[l], levels[l+1], xSize[l], ySize[l], zSize[l], xSize[l+1], ySize[l+1], zSize[l+1], l == levelMin, 0);
    }
    waitForCuda();
    t.stop_timer();
};
