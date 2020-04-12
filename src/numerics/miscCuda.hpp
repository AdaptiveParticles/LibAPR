//
// Created by joel on 07.04.20.
//

#ifndef LIBAPR_MISCCUDA_HPP
#define LIBAPR_MISCCUDA_HPP

#include "data_structures/APR/access/GPUAccess.hpp"
#include <cstdint>
#include <cstdio>

#ifdef APR_USE_CUDA
#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"
#endif

#define error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
#if defined(DEBUG) || defined(_DEBUG)
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
#endif
}


#define LOCALPATCHCONV333_N(particle_output,index,z,x,y,neighbour_sum)\
neighbour_sum=0;\
if (not_ghost) {\
    for (int q = 0; q < 3; ++q) {\
neighbour_sum  +=  local_stencil[q][0][0]*local_patch[z + q - 1][x + 0 - 1][(y+N-1)%N]\
                 + local_stencil[q][0][1]*local_patch[z + q - 1][x + 0 - 1][(y+N)%N]\
                 + local_stencil[q][0][2]*local_patch[z + q - 1][x + 0 - 1][(y+N+1)%N]\
                 + local_stencil[q][1][0]*local_patch[z + q - 1][x + 1 - 1][(y+N-1)%N]\
                 + local_stencil[q][1][1]*local_patch[z + q - 1][x + 1 - 1][(y+N)%N]\
                 + local_stencil[q][1][2]*local_patch[z + q - 1][x + 1 - 1][(y+N+1)%N]\
                 + local_stencil[q][2][0]*local_patch[z + q - 1][x + 2 - 1][(y+N-1)%N]\
                 + local_stencil[q][2][1]*local_patch[z + q - 1][x + 2 - 1][(y+N)%N]\
                 + local_stencil[q][2][2]*local_patch[z + q - 1][x + 2 - 1][(y+N+1)%N];\
    }\
    particle_output[index] = neighbour_sum;\
}\


#define LOCALPATCHCONV555_N(particle_output,index,z,x,y,neighbour_sum)\
neighbour_sum=0;\
if (not_ghost) {\
for (int q = 0; q < 5; ++q) {\
            neighbour_sum +=\
                local_stencil[q][0][0]*local_patch[z + q - 2][x + 0 - 2][(y+N-2)%N]\
                 + local_stencil[q][0][1]*local_patch[z + q - 2][x + 0 - 2][(y+N-1)%N]\
                 + local_stencil[q][0][2]*local_patch[z + q - 2][x + 0 - 2][(y+N)%N]\
                 + local_stencil[q][0][3]*local_patch[z + q - 2][x + 0 - 2][(y+N+1)%N]\
                 + local_stencil[q][0][4]*local_patch[z + q - 2][x + 0 - 2][(y+N+2)%N]\
                 + local_stencil[q][1][0]*local_patch[z + q - 2][x + 1 - 2][(y+N-2)%N]\
                 + local_stencil[q][1][1]*local_patch[z + q - 2][x + 1 - 2][(y+N-1)%N]\
                 + local_stencil[q][1][2]*local_patch[z + q - 2][x + 1 - 2][(y+N)%N]\
                 + local_stencil[q][1][3]*local_patch[z + q - 2][x + 1 - 2][(y+N+1)%N]\
                 + local_stencil[q][1][4]*local_patch[z + q - 2][x + 1 - 2][(y+N+2)%N]\
                 + local_stencil[q][2][0]*local_patch[z + q - 2][x + 2 - 2][(y+N-2)%N]\
                 + local_stencil[q][2][1]*local_patch[z + q - 2][x + 2 - 2][(y+N-1)%N]\
                 + local_stencil[q][2][2]*local_patch[z + q - 2][x + 2 - 2][(y+N)%N]\
                 + local_stencil[q][2][3]*local_patch[z + q - 2][x + 2 - 2][(y+N+1)%N]\
                 + local_stencil[q][2][4]*local_patch[z + q - 2][x + 2 - 2][(y+N+2)%N]\
                 + local_stencil[q][3][0]*local_patch[z + q - 2][x + 3 - 2][(y+N-2)%N]\
                 + local_stencil[q][3][1]*local_patch[z + q - 2][x + 3 - 2][(y+N-1)%N]\
                 + local_stencil[q][3][2]*local_patch[z + q - 2][x + 3 - 2][(y+N)%N]\
                 + local_stencil[q][3][3]*local_patch[z + q - 2][x + 3 - 2][(y+N+1)%N]\
                 + local_stencil[q][3][4]*local_patch[z + q - 2][x + 3 - 2][(y+N+2)%N]\
                 + local_stencil[q][4][0]*local_patch[z + q - 2][x + 4 - 2][(y+N-2)%N]\
                 + local_stencil[q][4][1]*local_patch[z + q - 2][x + 4 - 2][(y+N-1)%N]\
                 + local_stencil[q][4][2]*local_patch[z + q - 2][x + 4 - 2][(y+N)%N]\
                 + local_stencil[q][4][3]*local_patch[z + q - 2][x + 4 - 2][(y+N+1)%N]\
                 + local_stencil[q][4][4]*local_patch[z + q - 2][x + 4 - 2][(y+N+2)%N];\
}\
particle_output[index] = neighbour_sum;\
}\


template<typename T>
__global__ void elementWiseMult(T* in1, const T* in2, const size_t size);

template<typename T, typename S>
__global__ void elementWiseDiv(const T* numerator, const S* denominator, S* out, const size_t size);

template<typename T>
__global__ void copyKernel(const T* in, T* out, const size_t size);


template<typename T>
__global__ void fillWithValue(T* in, T value, const size_t size);


__global__ void print_value(const float* data, const size_t index);


template<unsigned int blockSize, typename T>
__global__ void compute_average(T* data, T* result, const size_t size);


template<int blockSize_z, int blockSize_x>
void compute_ne_rows_cuda(GPUAccessHelper& access, VectorData<int>& ne_count, ScopedCudaMemHandler<int*, JUST_ALLOC>& ne_rows, int blockSize);


void compute_ne_rows(GPUAccessHelper& access, VectorData<int>& ne_counter, VectorData<int>& ne_rows, int block_size);


#endif //LIBAPR_MISCCUDA_HPP
