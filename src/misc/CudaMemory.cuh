//
// Created by gonciarz on 8/8/18.
//

#ifndef LIBAPR_CUDAMEMORY_HPP
#define LIBAPR_CUDAMEMORY_HPP

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>


// TODO: this method is duplicated in CudaTools.cuh
//       Somehow including it here break compilation - fix it please.
#define checkCuda(ans) { cudaAssert2((ans), __FILE__, __LINE__); }
inline void cudaAssert2(cudaError_t code, const char *file, int line, bool abort=true)
{
#if defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: (%d) %s %s %d\n", code, cudaGetErrorString(code), file, line);
        assert(code == cudaSuccess); // If debugging it helps to see call tree somehow
        if (abort) exit(code);
    }
#endif
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
