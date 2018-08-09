#include "CudaMemory.hpp"
#include <cassert>

inline cudaError_t checkCuda(cudaError_t result) {
//#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
//#endif
    return result;
}

template <typename T>
inline T* getPinnedMemory(size_t aNumOfBytes) {
    T *memory = nullptr;
    cudaError_t result = checkCuda(cudaMallocHost((void**)&memory, aNumOfBytes) );
    std::cout << "Allocating pinned memory " << aNumOfBytes << " at " << (void*)memory << " result " << result << std::endl;
    return memory;
};

template <typename T>
inline void freePinnedMemory(T *aMemory) {
    std::cout << "Freeing pinned memory " << (void*)aMemory << std::endl;
    cudaFreeHost(aMemory);
}

template void freePinnedMemory(char*);
template void freePinnedMemory(uint8_t*);
template void freePinnedMemory(uint16_t*);
template void freePinnedMemory(short*);
template void freePinnedMemory(int*);
template void freePinnedMemory(float*);

template char* getPinnedMemory(size_t);
template uint8_t* getPinnedMemory(size_t);
template uint16_t* getPinnedMemory(size_t);
template short* getPinnedMemory(size_t);
template int* getPinnedMemory(size_t);
template float* getPinnedMemory(size_t);