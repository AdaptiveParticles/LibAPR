//
// Created by gonciarz on 8/8/18.
//

#ifndef LIBAPR_CUDAMEMORY_HPP
#define LIBAPR_CUDAMEMORY_HPP

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

inline void* getPinnedMemory(size_t aNumOfBytes) {
    void *memory = nullptr;
//    cudaError_t result =
    checkCuda(cudaMallocHost(&memory, aNumOfBytes) );
//    std::cout << "Allocating pinned memory " << aNumOfBytes << " at " << memory << " result " << result << std::endl;
    return memory;
}

inline void freePinnedMemory(void *aMemory) {
//    std::cout << "Freeing pinned memory " << aMemory << std::endl;
    cudaFreeHost(aMemory);
}

// useful extension of unique_ptr - with custom deleter there is no nice constructor taking just one parameter with memory
// here it is fixed with freePinnedMemory always passed nicely
template <typename T, typename D=decltype(&freePinnedMemory)>
struct PinnedMemoryUniquePtr : public std::unique_ptr<T[], D> {
    using std::unique_ptr<T[],D>::unique_ptr; // inheriting other constructors
    explicit PinnedMemoryUniquePtr(T *aMemory = nullptr) : std::unique_ptr<T[], D>(aMemory, &freePinnedMemory) {}
};

#endif //LIBAPR_CUDAMEMORY_HPP
