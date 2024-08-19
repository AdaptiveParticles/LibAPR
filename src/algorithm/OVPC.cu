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
__global__ void firstStep(T *data, size_t xLen, size_t yLen, size_t zLen, int level) {
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

    bool hasNeighHigherLevel = false;
    bool hasNeighSameLevel = false;
    for (int z = zmin; z <= zmax; ++z) {
        for (int x = xmin; x <= xmax; ++x) {
            for (int y = ymin; y <= ymax; ++y) {
                const size_t idx = z * xLen * yLen + x * yLen + y;
                T currentLevel = ~OVPC::MASK & data[idx];
                if (currentLevel > level) { hasNeighHigherLevel = true; break; }
                else if (currentLevel == level) hasNeighSameLevel = true;
            }
        }
    }
    if (!hasNeighHigherLevel) {
        const size_t idx = zi * xLen * yLen + xi * yLen + yi;
        T status = data[idx];
        if (status == level) data[idx] |= OVPC::SEED;
        else if (hasNeighSameLevel) data[idx] |= OVPC::BOUNDARY;
        else data[idx] |= OVPC::FILLER;
    }
}

template <typename T>
void runFirstStep(T *data, size_t xLen, size_t yLen, size_t zLen, int level, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 128, 1);
    dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (yLen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (zLen + threadsPerBlock.z - 1) / threadsPerBlock.z);
    firstStep<<<numBlocks,threadsPerBlock, 0, aStream>>>(data, xLen, yLen, zLen, level);
};

template <typename T>
__global__ void secondStep(T *data, T *child, size_t xLen, size_t yLen, size_t zLen, size_t xLenc, size_t yLenc, size_t zLenc, bool isLevelMin) {
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
    if (isLevelMin) data[zi * xLen * yLen + xi * yLen + yi] = status >> OVPC::BIT_SHIFT;
}

template <typename T>
void runSecondStep(T *data, T *child, size_t xLen, size_t yLen, size_t zLen, size_t xLenc, size_t yLenc, size_t zLenc, bool isLevelMax, cudaStream_t aStream) {
    dim3 threadsPerBlock(1, 128, 1);
    dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (yLen + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (zLen + threadsPerBlock.z - 1) / threadsPerBlock.z);
    secondStep<<<numBlocks,threadsPerBlock, 0, aStream>>>(data, child, xLen, yLen, zLen, xLenc, yLenc, zLenc, isLevelMax);
};

class ParticleCellTreeCuda {
    ScopedCudaMemHandler<uint8_t*, JUST_ALLOC> mem;
    std::vector<size_t> startOffsets;
    GenInfo gi;
    size_t numOfElements = 0;
    cudaStream_t stream = nullptr;

public:

    ParticleCellTreeCuda(const GenInfo &aprInfo, const cudaStream_t aStream) : gi(aprInfo), stream(aStream) {
        // Calculate size of needed memory for PCT and offsets for particular levels
        int l_max = aprInfo.l_max - 1;
        int l_min = aprInfo.l_min;

        startOffsets.resize(l_max + 1, 0);

        for (int l = l_min; l <= l_max; ++l) {
            auto yLen = ceil(aprInfo.org_dims[0] / PullingScheme::powr(2.0, l_max - l + 1));
            auto xLen = ceil(aprInfo.org_dims[1] / PullingScheme::powr(2.0, l_max - l + 1));
            auto zLen = ceil(aprInfo.org_dims[2] / PullingScheme::powr(2.0, l_max - l + 1));
            size_t levelSize = yLen * xLen * zLen;
            startOffsets[l] = numOfElements;
            numOfElements += levelSize;
        }

        // Initialize memory, it is not binded to any CPU memory so we provide nullptr
        mem.initialize(nullptr, numOfElements, stream);
        cudaMemsetAsync(mem.get(), EMPTY, numOfElements, stream);
    }

    inline uint8_t* operator[](size_t level) { return mem.get() + startOffsets[level]; }

    auto getPCTcpu() {
        std::vector<PixelData<uint8_t>> pct = PullingScheme::generateParticleCellTree(gi);
        for (int i = gi.l_min; i < gi.l_max; ++i) {
            checkCuda(cudaMemcpyAsync(pct[i].mesh.get(), (*this)[i], pct[i].mesh.size(), cudaMemcpyDeviceToHost, stream));
        }
        checkCuda(cudaStreamSynchronize(stream));

        return pct;
    }
};


// explicit instantiation of handled types
template std::vector<PixelData<uint8_t>> computeOvpcCuda(const PixelData<float>&, const GenInfo&);
template std::vector<PixelData<uint8_t>> computeOvpcCuda(const PixelData<int>&, const GenInfo&);

/**
 * CUDA implementation of Pullin Scheme (OVPC - Optimal Valid Particle Cell set).
 * @tparam T - type of input levels
 * @tparam S - type of output Particle Cell Tree
 * @param input - input levels computed in earlier stages
 * @param pct - Particle Cell Tree - as input is used for dimensions of each level, will be filled with computed
 *              Pulling Scheme as a output
 * @param levelMin - min level of APR
 * @param levelMax - max level of APR
 */
template <typename T>
std::vector<PixelData<uint8_t>> computeOvpcCuda(const PixelData<T> &input, const GenInfo &gi) {
    // Copy input to CUDA mem and prepare CUDA representation of particle cell tree which will be filled after computing
    // all steps

    ParticleCellTreeCuda pct(gi, 0 /*stream*/);
    int levelMin = gi.l_min;
    int levelMax = gi.l_max - 1;

    ScopedCudaMemHandler<const PixelData<T>, H2D> in(input);

    // feel the highes level of PCT with provided levels and clamp values to be within [levelMin, levelMax] range
    runCopyAndClampLevels(in.get(), pct[levelMax], in.getSize(), levelMin, levelMax, 0);

    // Downsample with max reduction to levelMin to fill the rest of the tree
    for (int l = levelMax - 1; l >= levelMin; --l) {
        runDownsampleMax(pct[l + 1], pct[l], gi.x_num[l + 1], gi.y_num[l + 1], gi.z_num[l + 1], 0);
    }

    // ================== Phase 1 - top to down
    for (int l = levelMin; l <= levelMax; ++l) {
        runFirstStep(pct[l], gi.x_num[l], gi.y_num[l], gi.z_num[l], l, 0);
    }
    // ================== Phase 1 - down to top
    for (int l = levelMax - 1; l >= levelMin; --l) {
        runSecondStep(pct[l], pct[l+1], gi.x_num[l], gi.y_num[l], gi.z_num[l], gi.x_num[l + 1], gi.y_num[l + 1], gi.z_num[l + 1], l == levelMin, 0);
    }

    return pct.getPCTcpu();
}
