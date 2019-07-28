//
// Created by cheesema on 2019-07-08.
//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "misc/CudaTools.cuh"
#include "misc/CudaMemory.cuh"
#include <chrono>
#include <cstdint>

#include "GPUAPR.hpp"

#define error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
#ifdef DEBUGCUDA
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
#endif
}

//template<int chunkSize, typename inputType, typename outputType>
//__global__ void iterate_ds_333(const uint64_t* level_xz_vec,
//                                const uint64_t* xz_end_vec,
//                                const uint16_t* y_vec,
//                                const inputType* input_particles,
//                                const uint64_t* level_xz_vec_tree,
//                                const uint64_t* xz_end_vec_tree,
//                                const uint16_t* y_vec_tree,
//                                outputType* output_particles,
//                                const int z_num,
//                                const int x_num,
//                                const int y_num,
//                                const int z_num_parent,
//                                const int x_num_parent,
//                                const int y_num_parent,
//                                const int level) {
//
//    /// assumes blockDim = (3, chunkSize, 3)
//    const int x_index = blockIdx.x + threadIdx.x - 1;
//    const int z_index = blockIdx.z + threadIdx.z - 1;
//
//    const int row = threadIdx.x + threadIdx.z * blockDim.x;
//    //const int local_th = threadIdx.x % 9;
//
//    __shared__ size_t global_index_begin_0_s[9];
//    __shared__ size_t global_index_end_0_s[9];
//
//    if( (x_index >= x_num) || (z_index >= z_num) ) {
//        return;
//    } else {
//        if(threadIdx.y == 0) {
//            size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
//            global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
//            global_index_end_0_s[row] = xz_end_vec[xz_start];
//        }
//    }
//    __syncthreads();
//
//    if(global_index_begin_0_s[row] == global_index_end_0_s[row]){
//        return;
//    }
//
//    const size_t global_index_begin_0 = global_index_begin_0_s[row];
//    const size_t global_index_end_0 = global_index_end_0_s[row];
//
//    __shared__ float patch_cache[9][chunkSize];
//
//    patch_cache[row][threadIdx.y] = 0;
//
//    float current_val = 0;
//    int current_y = -1;
//
//    if((global_index_begin_0 + threadIdx.y) < global_index_end_0) {
//        current_val = input_particles[global_index_begin_0 + threadIdx.y];
//        current_y = y_vec[global_index_begin_0 + threadIdx.y];
//    }
//
//    int sparse_block = 0;
//    const int number_y_chunk = (y_num + chunkSize) / chunkSize;
//
//    for(int y_chunk = 0; y_chunk < number_y_chunk; ++y_chunk) {
//
//        __syncthreads();
//        if(current_y < y_chunk*chunkSize) {
//            sparse_block++;
//            if( (sparse_block*chunkSize + global_index_begin_0 + threadIdx.y) < global_index_end_0) {
//                current_val = input_particles[sparse_block*chunkSize + global_index_begin_0 + threadIdx.y];
//                current_y = y_vec[sparse_block*chunkSize + global_index_begin_0 + threadIdx.y];
//            }
//        }
//
//        if( (current_y >= (y_chunk*chunkSize)) && (current_y < ((y_chunk+1)*chunkSize)) ) {
//            patch_cache[row][current_y % chunkSize] = current_val;
//        }
//        __syncthreads();
//
//        if(row == 4) {
//
//
//
//        }
//
//    }
//}

template<int numRows, int chunkSize, typename inputType, typename outputType>
__global__ void chunked_iterate(const uint64_t* level_xz_vec,
                               const uint64_t* xz_end_vec,
                               const uint16_t* y_vec,
                               const inputType* input_particles,
                               outputType* output_particles,
                               const int z_num,
                               const int x_num,
                               const int y_num,
                               const int level) {

    /// assumes blockDim = (nx, chunkSize, nz),  numRows == nx * nz
    const int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int z_index = blockIdx.z * blockDim.z + threadIdx.z;

    const int row = threadIdx.x + threadIdx.z * blockDim.x;
    //const int local_th = threadIdx.x % 9;

    __shared__ size_t global_index_begin_0_s[numRows];
    __shared__ size_t global_index_end_0_s[numRows];

    if( (x_index >= x_num) || (z_index >= z_num) ) {
        return;
    } else {
        if(threadIdx.y == 0) {
            size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
            global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
            global_index_end_0_s[row] = xz_end_vec[xz_start];
        }
    }
    __syncthreads();

    if(global_index_begin_0_s[row] == global_index_end_0_s[row]){
        return;
    }

    const size_t global_index_begin_0 = global_index_begin_0_s[row];
    const size_t global_index_end_0 = global_index_end_0_s[row];

    float current_val = 0;
    int current_y = -1;

    if((global_index_begin_0 + threadIdx.y) < global_index_end_0) {
        current_val = input_particles[global_index_begin_0 + threadIdx.y];
        current_y = y_vec[global_index_begin_0 + threadIdx.y];
    }

    const int chunk_start = y_vec[global_index_begin_0] / chunkSize;
    const int chunk_end = fminf((int) y_vec[global_index_end_0]/chunkSize + 2, (y_num + chunkSize - 1) / chunkSize);

    int sparse_block = 0;

    for(int y_chunk = chunk_start; y_chunk < chunk_end; ++y_chunk) {

        if(current_y < y_chunk*chunkSize) {
            sparse_block++;
            if( (sparse_block*chunkSize + global_index_begin_0 + threadIdx.y) < global_index_end_0) {
                current_val = input_particles[sparse_block*chunkSize + global_index_begin_0 + threadIdx.y];
                current_y = y_vec[sparse_block*chunkSize + global_index_begin_0 + threadIdx.y];
            }
        }

        if( (sparse_block*chunkSize + global_index_begin_0 + threadIdx.y) < global_index_end_0 ) {
            output_particles[sparse_block*chunkSize + global_index_begin_0 + threadIdx.y] =
                    z_index + x_index + current_y + level;
        }
    }
}


template <typename inputType, typename outputType>
__global__ void sequential_iterate(const uint64_t* level_xz_vec,
                                   const uint64_t* xz_end_vec,
                                   const uint16_t* y_vec,
                                   const inputType* input_particles,
                                   outputType* output_particles,
                                   const int z_num,
                                   const int x_num,
                                   const int y_num,
                                   const int level) {

    const int x_idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int z_idx = (size_t)blockDim.z * blockIdx.z + threadIdx.z;

    if(x_idx >= x_num) { return; }
    if(z_idx >= z_num) { return; }

    uint64_t xz_start = x_idx + z_idx * x_num + level_xz_vec[level];

    uint64_t begin_index = xz_end_vec[xz_start-1];
    uint64_t end_index = xz_end_vec[xz_start];
    uint16_t y_idx;
    float current_val = 0;

    for(uint64_t i = begin_index; i < end_index; ++i) {
        current_val = input_particles[i];
        y_idx = y_vec[i];

        output_particles[i] = z_idx + x_idx + y_idx + level;
    }
}


template<int blockSize, int nthreads>
double bench_chunked_iterate(GPUAccessHelper& access, int num_rep) {

    APRTimer timer(false);

    std::vector<uint16_t> input(access.total_number_particles(), 0);
    std::vector<uint16_t> output(access.total_number_particles(), 0);

    ScopedCudaMemHandler<uint16_t *, JUST_ALLOC> input_gpu(input.data(), access.total_number_particles());
    ScopedCudaMemHandler<uint16_t *, JUST_ALLOC> output_gpu(output.data(), access.total_number_particles());
    input_gpu.copyH2D();

    timer.start_timer("chunked_iteration");

    const int chunkSize = nthreads/(blockSize*blockSize);

    for (int r = 0; r < num_rep; ++r) {

        for (unsigned int level = access.level_max(); level >= access.level_min(); --level) {

            dim3 threadsPerBlock(blockSize, chunkSize, blockSize);

            const int x_blocks = (access.x_num(level) + threadsPerBlock.x - 1) / threadsPerBlock.x;
            const int z_blocks = (access.z_num(level) + threadsPerBlock.z - 1) / threadsPerBlock.z;
            dim3 numBlocks(x_blocks, 1, z_blocks);

            chunked_iterate<blockSize * blockSize, chunkSize> << < numBlocks, threadsPerBlock >> >
                                                                              (access.get_level_xz_vec_ptr(),
                                                                                      access.get_xz_end_vec_ptr(),
                                                                                      access.get_y_vec_ptr(),
                                                                                      input_gpu.get(),
                                                                                      output_gpu.get(),
                                                                                      access.z_num(level),
                                                                                      access.x_num(level),
                                                                                      access.y_num(level),
                                                                                      level);

            error_check( cudaPeekAtLastError() )
        }
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    double avg_time = timer.timings.back() / num_rep;

    return avg_time;
}

template double bench_chunked_iterate<1, 128>(GPUAccessHelper&, int);
template double bench_chunked_iterate<2, 128>(GPUAccessHelper&, int);
template double bench_chunked_iterate<4, 128>(GPUAccessHelper&, int);
template double bench_chunked_iterate<8, 128>(GPUAccessHelper&, int);

template double bench_chunked_iterate<1, 256>(GPUAccessHelper&, int);
template double bench_chunked_iterate<2, 256>(GPUAccessHelper&, int);
template double bench_chunked_iterate<4, 256>(GPUAccessHelper&, int);
template double bench_chunked_iterate<8, 256>(GPUAccessHelper&, int);


template<int blockSize>
double bench_sequential_iterate(GPUAccessHelper& access, int num_rep) {

    APRTimer timer(false);

    std::vector<uint16_t> input(access.total_number_particles(), 0);
    std::vector<uint16_t> output(access.total_number_particles(), 0);

    ScopedCudaMemHandler<uint16_t *, JUST_ALLOC> input_gpu(input.data(), access.total_number_particles());
    ScopedCudaMemHandler<uint16_t *, JUST_ALLOC> output_gpu(output.data(), access.total_number_particles());
    input_gpu.copyH2D();

    timer.start_timer("chunked_iteration");

    for(int r = 0; r < num_rep; ++r) {

        for (unsigned int level = access.level_max(); level >= access.level_min(); --level) {

            dim3 threadsPerBlock(blockSize, 1, blockSize);

            const int x_blocks = (access.x_num(level) + threadsPerBlock.x - 1) / threadsPerBlock.x;
            const int z_blocks = (access.z_num(level) + threadsPerBlock.z - 1) / threadsPerBlock.z;
            dim3 numBlocks(x_blocks, 1, z_blocks);

            sequential_iterate<<< numBlocks, threadsPerBlock >>>
                                                     (access.get_level_xz_vec_ptr(),
                                                      access.get_xz_end_vec_ptr(),
                                                      access.get_y_vec_ptr(),
                                                      input_gpu.get(),
                                                      output_gpu.get(),
                                                      access.z_num(level),
                                                      access.x_num(level),
                                                      access.y_num(level),
                                                      level);

            error_check( cudaPeekAtLastError() )
        }
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    double avg_time = timer.timings.back() / num_rep;

    return avg_time;
}

template double bench_sequential_iterate<1>(GPUAccessHelper&, int);
template double bench_sequential_iterate<2>(GPUAccessHelper&, int);
template double bench_sequential_iterate<4>(GPUAccessHelper&, int);
template double bench_sequential_iterate<8>(GPUAccessHelper&, int);
template double bench_sequential_iterate<16>(GPUAccessHelper&, int);



void compute_spatial_info_gpu(GPUAccessHelper& access, std::vector<uint16_t>& input, std::vector<uint16_t>& output){

    input.resize(access.total_number_particles(), 0);
    output.resize(access.total_number_particles(), 0);

    ScopedCudaMemHandler<uint16_t*, JUST_ALLOC> input_gpu(input.data(), access.total_number_particles());
    ScopedCudaMemHandler<uint16_t*, JUST_ALLOC> output_gpu(output.data(), access.total_number_particles());
    input_gpu.copyH2D();

    std::cout << "gpu size: " << input_gpu.getSize() << " cpu size: " << input.size() << std::endl;

    const int blockSize = 2;
    const int chunkSize = 128 / (blockSize*blockSize);

    for( unsigned int level = access.level_max(); level >= access.level_min(); --level) {

        dim3 threadsPerBlock(blockSize, chunkSize, blockSize);

        const int x_blocks = (access.x_num(level) + threadsPerBlock.x - 1) / threadsPerBlock.x;
        const int z_blocks = (access.z_num(level) + threadsPerBlock.z - 1) / threadsPerBlock.z;
        dim3 numBlocks(x_blocks, 1, z_blocks);

        chunked_iterate<blockSize*blockSize, chunkSize><<< numBlocks, threadsPerBlock >>>
                (access.get_level_xz_vec_ptr(),
                access.get_xz_end_vec_ptr(),
                access.get_y_vec_ptr(),
                input_gpu.get(),
                output_gpu.get(),
                access.z_num(level),
                access.x_num(level),
                access.y_num(level),
                level);

        error_check( cudaPeekAtLastError() )
    }
    output_gpu.copyD2H();
}


template <typename T>
__global__ void simpleKernel(T *input, uint64_t size) {

    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= size) { return; }

    input[idx] = idx;
}

void run_simple_test(std::vector<uint64_t>& temp, uint64_t size) {

    temp.resize(size, 0);

    ScopedCudaMemHandler<uint64_t*, H2D> temp_gpu(temp.data(), size);
    temp_gpu.copyH2D();

    error_check( cudaPeekAtLastError() )

    std::cout << "gpu size: " << temp_gpu.getSize() << " cpu size: " << temp.size() << std::endl;

    //cudaStream_t stream;
    //cudaStreamCreate(&stream);

    dim3 threadsPerBlock(64);
    dim3 numBlocks((temp.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);

    simpleKernel << < numBlocks, threadsPerBlock >> > (temp_gpu.get(), size);

    error_check( cudaPeekAtLastError() )

    temp_gpu.copyD2H();

    error_check( cudaPeekAtLastError() )
}


template <typename T, typename S>
__global__ void copyKernel(T *input, S* output, uint64_t size) {

    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= size) { return; }

    output[idx] = (S) input[idx];
}

void check_access_vectors(GPUAccessHelper& access, std::vector<uint16_t>& y_vec_out, std::vector<uint64_t>& xz_end_vec_out, std::vector<uint64_t>& level_xz_vec_out) {

    y_vec_out.resize(access.linearAccess->y_vec.size(), 0);
    xz_end_vec_out.resize(access.linearAccess->xz_end_vec.size(), 0);
    level_xz_vec_out.resize(access.linearAccess->level_xz_vec.size(), 0);

    for(uint64_t i = 0; i < level_xz_vec_out.size(); ++i) {
        level_xz_vec_out[i] = i;
    }


    ScopedCudaMemHandler<uint16_t*, H2D> check_array_y(y_vec_out.data(), y_vec_out.size());
    ScopedCudaMemHandler<uint64_t*, H2D> check_array_xz(xz_end_vec_out.data(), xz_end_vec_out.size());
    ScopedCudaMemHandler<uint64_t*, H2D> check_array_lvl(level_xz_vec_out.data(), level_xz_vec_out.size());

    check_array_y.copyH2D();
    check_array_xz.copyH2D();
    check_array_lvl.copyH2D();

    error_check( cudaPeekAtLastError() )

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threadsPerBlock(64);
    dim3 numBlocks((level_xz_vec_out.size() + threadsPerBlock.x - 1) / threadsPerBlock.x);

    copyKernel<< < numBlocks, threadsPerBlock >> > (access.get_level_xz_vec_ptr(), check_array_lvl.get(), level_xz_vec_out.size());

    error_check( cudaPeekAtLastError() )

    numBlocks.x = (xz_end_vec_out.size() + threadsPerBlock.x - 1) / threadsPerBlock.x;
    copyKernel<< < numBlocks, threadsPerBlock >> > (access.get_xz_end_vec_ptr(), check_array_xz.get(), xz_end_vec_out.size());

    error_check( cudaPeekAtLastError() )

    numBlocks.x = (y_vec_out.size() + threadsPerBlock.x - 1) / threadsPerBlock.x;
    copyKernel<< < numBlocks, threadsPerBlock >> > (access.get_y_vec_ptr(), check_array_y.get(), y_vec_out.size());

    error_check( cudaPeekAtLastError() )

    check_array_y.copyD2H();
    check_array_xz.copyD2H();
    check_array_lvl.copyD2H();

    error_check( cudaPeekAtLastError() )
}
