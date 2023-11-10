//
// Created by joel on 07.04.20.
//
#include "miscCuda.hpp"

/**
 * Element-wise multiplication of two vectors
 *
 * @tparam T        data type
 * @param in1       first vector ( this is where the output will be stored )
 * @param in2       second vector
 * @param size      size of the input vectors
 */
template<typename T>
__global__ void elementWiseMult(T* in1,
                                const T* in2,
                                const size_t size){

    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        in1[idx] = in1[idx] * in2[idx];
    }
}


/**
 * Element-wise division of two vectors.
 *
 * @tparam T
 * @param numerator   numerator
 * @param denominator   denominator
 * @param out   output
 * @param size  size of the vectors
 */
template<typename T, typename S>
__global__ void elementWiseDiv(const T* numerator,
                               const S* denominator,
                               S* out,
                               const size_t size){

    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        out[idx] = ((S) numerator[idx]) / denominator[idx];
    }
}


/**
 * Add the square of one vector to another. For 0 <= idx < size:
 *      in1[idx] += in2[idx] * in2[idx];
 */
__global__ void addSquare(float* in1,
                          const float* in2,
                          const size_t size) {

    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        in1[idx] += in2[idx] * in2[idx];
    }
}


/**
 * Take the square root of each element in the input vector
 */
__global__ void elementWiseSqrt(float* __restrict__ input,
                                const size_t size) {

    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        input[idx] = sqrtf(input[idx]);
    }
}


template<typename T, typename S>
__global__ void copyKernel(const T* __restrict__ in, S* __restrict__ out, const size_t size){

    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        out[idx] = in[idx];
    }
}

template<typename T>
__global__ void fillWithValue(T* in, T value, const size_t size){

    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        in[idx] = value;
    }
}

template<unsigned int blockSize, typename T>
__global__ void compute_average(T* data, T* result, const size_t size) {

    /**
     * Assumes a single 1-dimensional cuda block of size (blockSize, 1, 1) !
     */

    double local_sum = 0;

    __shared__ double sdata[blockSize];

    for(size_t i = threadIdx.x; i < size; i += blockSize) {
        local_sum += data[i];
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // sum reduction
    for(int s = 1; s < blockSize; s *= 2) {
        if(threadIdx.x % (2*s) == 0) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        *result = (T) (sdata[0] / size);
    }
}

__global__ void print_value(const float* data, const size_t index) {
    printf("data[%d] = %f\n", (int)index, data[index]);
}


__forceinline__ __device__ int block_nonempty(const uint64_t* level_xz_vec,
                                              const uint64_t* xz_end_vec,
                                              const int x_num,
                                              const int level,
                                              const int x_index,
                                              const int x_limit,
                                              const int z_index,
                                              const int z_limit) {

    for(int iz = 0; iz < z_limit; ++iz) {
        for(int ix = 0; ix < x_limit; ++ix) {
            size_t xz_start = (z_index + iz) * x_num + (x_index + ix) + level_xz_vec[level];

            // if row is non-empty
            if( xz_end_vec[xz_start - 1] < xz_end_vec[xz_start]) {
                return 1;
            }
        }
    }
    return 0;
}

template<int blockSize_z, int blockSize_x>
__global__ void count_ne_rows_cuda(const uint64_t* level_xz_vec,
                                   const uint64_t* xz_end_vec,
                                   const int z_num,
                                   const int x_num,
                                   const int level,
                                   const int chunkSize,
                                   int* res) {

    __shared__ int local_counts[blockSize_x][blockSize_z];
    local_counts[threadIdx.y][threadIdx.x] = 0;

    const int z_index = blockIdx.x * blockDim.x * chunkSize + threadIdx.x * chunkSize;

    if(z_index >= z_num) { return; } // out of bounds

    int x_index = threadIdx.y * chunkSize;

    int counter = 0;
    const int z_limit = (z_index < z_num-chunkSize) ? chunkSize : z_num-z_index;


    // loop over x-dimension in chunks
    while( x_index < x_num ) {

        const int x_limit = (x_index < x_num-chunkSize) ? chunkSize : x_num-x_index;

        int nonempty = block_nonempty(level_xz_vec, xz_end_vec, x_num, level, x_index, x_limit, z_index, z_limit);
        counter += nonempty;

        x_index += blockDim.y * chunkSize;
    }
    __syncthreads();

    local_counts[threadIdx.y][threadIdx.x] = counter;
    __syncthreads();

    // reduce over blockDim.y to get the count for each z_index
    for(int gap = blockSize_x/2; gap > 0; gap/=2) {
        if(threadIdx.y < gap) {
            local_counts[threadIdx.y][threadIdx.x] += local_counts[threadIdx.y + gap][threadIdx.x];
        }
        __syncthreads();
    }

    // now reduce over blockDim.x to get the block count
    for(int gap = blockSize_z/2; gap > 0; gap/=2) {
        if(threadIdx.x < gap && threadIdx.y == 0) {
            local_counts[0][threadIdx.x] += local_counts[0][threadIdx.x + gap];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        res[blockIdx.x] = local_counts[0][0];
    }
}


__device__ unsigned int count = 0;
__global__ void fill_ne_rows_cuda(const uint64_t* level_xz_vec,
                                  const uint64_t* xz_end_vec,
                                  const int z_num,
                                  const int x_num,
                                  const int level,
                                  const int chunkSize,
                                  unsigned int ne_count,
                                  int offset,
                                  int* ne_rows) {

    const int z_index = blockIdx.x * blockDim.x * chunkSize + threadIdx.x * chunkSize;

    if (z_index >= z_num) { return; } // out of bounds

    int x_index = threadIdx.y * chunkSize;

    const int z_limit = (z_index < z_num - chunkSize) ? chunkSize : z_num - z_index;

    // loop over x-dimension in chunks
    while (x_index < x_num) {

        const int x_limit = (x_index < x_num - chunkSize) ? chunkSize : x_num - x_index;

        // if row is non-empty
        if(block_nonempty(level_xz_vec, xz_end_vec, x_num, level, x_index, x_limit, z_index, z_limit) ) {
            unsigned int index = atomicInc(&count, ne_count-1);
            ne_rows[offset + index] = z_index * x_num + x_index;
        }

        x_index += blockDim.y * chunkSize;
    }
}



template<int blockSize_z, int blockSize_x>
void compute_ne_rows_cuda(GPUAccessHelper& access, VectorData<int>& ne_count, ScopedCudaMemHandler<int*, JUST_ALLOC>& ne_rows_gpu, int blockSize) {

    ne_count.resize(access.level_max()+2, 0);

    int stride = blockSize_z * blockSize;

    int z_blocks_max = (access.z_num(access.level_max()) + stride - 1) / stride;
    int num_levels = access.level_max() - access.level_min() + 1;

    int block_sums_host[z_blocks_max * num_levels];
    int *block_sums_device;

    error_check(cudaMalloc(&block_sums_device, z_blocks_max*num_levels*sizeof(int)) )
    error_check( cudaMemset(block_sums_device, 0, z_blocks_max*num_levels*sizeof(int)) )

    int offset = 0;
    for(int level = access.level_min(); level <= access.level_max(); ++level) {

        int z_blocks = (access.z_num(level) + stride - 1) / stride;

        dim3 grid_dim(z_blocks, 1, 1);
        dim3 block_dim(blockSize_z, blockSize_x, 1);

        count_ne_rows_cuda<blockSize_z, blockSize_x>
                << < grid_dim, block_dim >> >
                               (access.get_level_xz_vec_ptr(),
                                access.get_xz_end_vec_ptr(),
                                access.z_num(level),
                                access.x_num(level),
                                level,
                                blockSize,
                                block_sums_device + offset);
        offset += z_blocks_max;
    }

    error_check(cudaDeviceSynchronize())
    error_check(cudaMemcpy(block_sums_host, block_sums_device, z_blocks_max*num_levels*sizeof(int), cudaMemcpyDeviceToHost) )

    int counter = 0;
    offset = 0;

    for(int level = access.level_min(); level <= access.level_max(); ++level) {
        ne_count[level] = counter;

        for(int i = 0; i < z_blocks_max; ++i) {
            counter += block_sums_host[offset + i];
        }

        offset += z_blocks_max;
    }

    ne_count.back() = counter;
    ne_rows_gpu.initialize(NULL, counter);

    for(int level = access.level_min(); level <= access.level_max(); ++level) {

        int ne_sz = ne_count[level+1] - ne_count[level];
        if( ne_sz == 0 ) {
            continue;
        }
        int z_blocks = (access.z_num(level) + blockSize_z * blockSize - 1) / (blockSize_z * blockSize);

        dim3 grid_dim(z_blocks, 1, 1);
        dim3 block_dim(blockSize_z, blockSize_x, 1);

        fill_ne_rows_cuda<<< grid_dim, block_dim >>>
                                       (access.get_level_xz_vec_ptr(),
                                        access.get_xz_end_vec_ptr(),
                                        access.z_num(level),
                                        access.x_num(level),
                                        level,
                                        blockSize,
                                        ne_sz,
                                        ne_count[level],
                                        ne_rows_gpu.get());
    }

    error_check( cudaFree(block_sums_device) )
}


/// CPU version of count nonempty rows. TODO: put this somewhere else

inline void count_nonempty(GPUAccessHelper& access, uint64_t& counter, const uint64_t level_start, const int level, const int z, const int x, const int block_size_z, const int block_size_x) {
    for(int iz = 0; iz < block_size_z; ++iz ){
        for(int ix = 0; ix < block_size_x; ++ix ) {
            auto offset = x + ix + (z + iz) * access.x_num(level);
            auto xz_start = level_start + offset;

            auto begin_index = access.linearAccess->xz_end_vec[xz_start - 1];
            auto end_index = access.linearAccess->xz_end_vec[xz_start];

            if(begin_index < end_index) {
                counter++;
                return;
            }
        }
    }
}


inline void add_nonempty(GPUAccessHelper& access, uint64_t& counter, VectorData<int>& ne_rows, const uint64_t level_start, const int level, const int z, const int x, const int block_size_z, const int block_size_x) {
    for(int iz = 0; iz < block_size_z; ++iz ){
        for(int ix = 0; ix < block_size_x; ++ix ) {
            auto offset = x + ix + (z + iz) * access.x_num(level);
            auto xz_start = level_start + offset;

            auto begin_index = access.linearAccess->xz_end_vec[xz_start - 1];
            auto end_index = access.linearAccess->xz_end_vec[xz_start];

            if(begin_index < end_index) {
                ne_rows[counter] = x + z * access.x_num(level);
                counter++;
                return;
            }
        }
    }
}


void compute_ne_rows(GPUAccessHelper& access, VectorData<int>& ne_counter, VectorData<int>& ne_rows, int block_size) {
    ne_counter.resize(access.level_max()+2, 0);

    int z = 0;
    int x = 0;

    uint64_t counter = 0;

    /// loop over the data structure to compute the number of nonempty blocks
    for (int level = access.level_min(); level <= access.level_max(); ++level) {

        auto level_start = access.linearAccess->level_xz_vec[level];

        ne_counter[level] = counter;

//#ifdef HAVE_OPENMP
//#pragma omp parallel for firstprivate(z, x) reduction(+: counter)
//#endif
        for (z = 0; z <= (access.z_num(level) - block_size); z+=block_size) {
            for (x = 0; x <= (access.x_num(level) - block_size); x+=block_size) {
                count_nonempty(access, counter, level_start, level, z, x, block_size, block_size);
            }

            if( x < access.x_num(level) ) {
                count_nonempty(access, counter, level_start, level, z, x, block_size, access.x_num(level) - x);
            }
        }

        if( z < access.z_num(level)) {
            int block_size_z = access.z_num(level) - z;

            for (x = 0; x <= (access.x_num(level) - block_size); x+=block_size) {
                count_nonempty(access, counter, level_start, level, z, x, block_size_z, block_size);
            }

            if( x < access.x_num(level) ) {
                count_nonempty(access, counter, level_start, level, z, x, block_size_z, access.x_num(level) - x);
            }
        }
    }

    ne_counter.back() = counter; // last value at level_max+1
    ne_rows.resize(counter);
    counter = 0;

    /// loop through again and add the offsets
    for (int level = access.level_min(); level <= access.level_max(); ++level) {

        auto level_start = access.linearAccess->level_xz_vec[level];

        ne_counter[level] = counter;

        for (z = 0; z <= (access.z_num(level) - block_size); z+=block_size) {
            for (x = 0; x <= (access.x_num(level) - block_size); x+=block_size) {
                add_nonempty(access, counter, ne_rows, level_start, level, z, x, block_size, block_size);
            }

            if( x < access.x_num(level) ) {
                add_nonempty(access, counter, ne_rows, level_start, level, z, x, block_size, access.x_num(level) - x);
            }
        }

        if( z < access.z_num(level)) {
            int block_size_z = access.z_num(level) - z;

            for (x = 0; x <= (access.x_num(level) - block_size); x+=block_size) {
                add_nonempty(access, counter, ne_rows, level_start, level, z, x, block_size_z, block_size);
            }

            if( x < access.x_num(level) ) {
                add_nonempty(access, counter, ne_rows, level_start, level, z, x, block_size_z, access.x_num(level) - x);
            }
        }
    }
}


template __global__ void elementWiseMult(float*, const float*, const size_t size);
template __global__ void elementWiseMult(uint16_t*, const uint16_t*, const size_t size);

template __global__ void elementWiseDiv(const float*, const float*, float*, const size_t);
template __global__ void elementWiseDiv(const uint16_t*, const float*, float*, const size_t);
template __global__ void elementWiseDiv(const uint8_t*, const float*, float*, const size_t);

template __global__ void copyKernel(const float*, float*, const size_t);
template __global__ void copyKernel(const uint16_t*, float*, const size_t);
template __global__ void copyKernel(const uint8_t*, float*, const size_t);

template __global__ void fillWithValue(float*, float, const size_t);
template __global__ void fillWithValue(uint16_t*, uint16_t, const size_t);

template __global__ void compute_average<512>(float*, float*, const size_t);
template __global__ void compute_average<512>(uint16_t*, uint16_t*, const size_t);

template void compute_ne_rows_cuda<8, 32>(GPUAccessHelper &, VectorData<int> &, ScopedCudaMemHandler<int*, JUST_ALLOC>&, int);
template void compute_ne_rows_cuda<16, 32>(GPUAccessHelper &, VectorData<int> &, ScopedCudaMemHandler<int*, JUST_ALLOC>&, int);
template void compute_ne_rows_cuda<32, 32>(GPUAccessHelper &, VectorData<int> &, ScopedCudaMemHandler<int*, JUST_ALLOC>&, int);
