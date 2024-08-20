#ifndef PARTICLE_CELL_TREE_CUDA_CUH
#define PARTICLE_CELL_TREE_CUDA_CUH


#include "data_structures/APR/GenInfo.hpp"
#include "algorithm/PullingScheme.hpp"


/*
 * CUDA representation of PCT (Particle Cell Tree)
 * Allocates memory and initialize it to EMPTY
 *
 * Allows acces to each level via subscription operator:
 * ParticleCellTreeCuda pct(aprInfo);
 * pct[level]
 *
 * getPCTcpu and uploadPCT2GPU handle interaction with CPU code (mainly for test/debug purposes).
 */
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

    void downloadPCTfromGPU(std::vector<PixelData<uint8_t>> &pct) {
        for (int i = gi.l_min; i < gi.l_max; ++i) {
            checkCuda(cudaMemcpyAsync(pct[i].mesh.get(), (*this)[i], pct[i].mesh.size(), cudaMemcpyDeviceToHost, stream));
        }
        checkCuda(cudaStreamSynchronize(stream));
    }

    void uploadPCT2GPU(const std::vector<PixelData<uint8_t>> &pct) {
        for (int i = gi.l_min; i < gi.l_max; ++i) {
            checkCuda(cudaMemcpyAsync((*this)[i], pct[i].mesh.get(), pct[i].mesh.size(), cudaMemcpyHostToDevice, stream));
        }
        checkCuda(cudaStreamSynchronize(stream));
    }
};


#endif
