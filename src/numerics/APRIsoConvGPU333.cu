//
// Created by joel on 07.04.20.
//

#include "APRIsoConvGPU333.hpp"

/// -------------------------------------------------------- ///
///                     with zero padding                    ///
/// -------------------------------------------------------- ///

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_chunked(const uint64_t* __restrict__ level_xz_vec,
                                     const uint64_t* __restrict__ xz_end_vec,
                                     const uint16_t* __restrict__ y_vec,
                                     const inputType* __restrict__ input_particles,
                                     outputType* __restrict__ output_particles,
                                     const stencilType* __restrict__ stencil,
                                     const int z_num,
                                     const int x_num,
                                     const int y_num,
                                     const int z_num_parent,
                                     const int x_num_parent,
                                     const int level,
                                     const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) { //out of bounds
        return;
    }

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (blockSize - 1)) &&
                           (threadIdx.z > 0) && (threadIdx.z < (blockSize - 1));

    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    const int x_index_p = x_index / 2;
    const int z_index_p = z_index / 2;

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }
    __syncthreads();

    if(global_index_begin_0_s[5] == global_index_end_0_s[5]) {
        return;
    }

    if(threadIdx.x == 0) {
        size_t xz_start = x_index_p + z_index_p * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    __syncthreads();

    inputType f_0, f_p;
    int y_0, y_p;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;

    if(update_index < global_index_end_0_s[row]) {
        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];
    } else {
        y_0 = INT32_MAX;
    }

    __syncwarp();

    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    // overlapping y chunks

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-2)*(blockSize-2)];
    __shared__ int chunk_start[(blockSize-2)*(blockSize-2)];

    __syncthreads();

    if((threadIdx.z == 1) && (threadIdx.y == 1) && (threadIdx.x < (blockSize-2)*(blockSize-2))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-2;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 1 + (threadIdx.z - 1)*(blockSize-2);
    for(int j = 1; j < ((blockSize-2)*(blockSize-2)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    int sparse_block = 0;
    int sparse_block_p = 0;
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();

        // update apr particle
        while( y_0 < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block++;

            if( (sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent particle
        while( y_p < (y_chunk*chunkSizeInternal - 1)) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if(y_0 <= (y_chunk+1)*chunkSizeInternal) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % chunkSize] = f_0;
        }

        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num)) {
            local_patch[threadIdx.z][threadIdx.y][(y_p+1) % chunkSize] = f_p;
        }

        __syncthreads();

        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 1, neighbour_sum)

        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_chunked(const uint64_t* __restrict__ level_xz_vec,
                                          const uint64_t* __restrict__ xz_end_vec,
                                          const uint16_t* __restrict__ y_vec,
                                          const inputType* __restrict__ input_particles,
                                          outputType* __restrict__ output_particles,
                                          const stencilType* __restrict__ stencil,
                                          const uint64_t* __restrict__ level_xz_vec_tree,
                                          const uint64_t* __restrict__ xz_end_vec_tree,
                                          const uint16_t* __restrict__ y_vec_tree,
                                          const treeType* __restrict__ tree_data,
                                          const int z_num,
                                          const int x_num,
                                          const int y_num,
                                          const int z_num_parent,
                                          const int x_num_parent,
                                          const int level,
                                          const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {
        return; // out of bounds
    }

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (blockSize - 1)) &&
                           (threadIdx.z > 0) && (threadIdx.z < (blockSize - 1));

    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_t_s[blockSize*blockSize];
    __shared__ size_t global_index_end_t_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    const int x_index_p = x_index / 2;
    const int z_index_p = z_index / 2;

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 1) {
        size_t xz_start = x_index_p + z_index_p * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 2) {
        size_t xz_start = level_xz_vec_tree[level] + (x_index) + (z_index) * x_num;
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];
    }

    __syncthreads();

    stencilType f_0, f_p, f_t;
    int y_0, y_p, y_t;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;

    if((update_index) < global_index_end_0_s[row]) {

        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];

    } else {
        y_0 = INT32_MAX;
    }

    __syncthreads();

    if((global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row]) {

        f_t = tree_data[global_index_begin_t_s[row] + threadIdx.x];
        y_t = y_vec_tree[global_index_begin_t_s[row] + threadIdx.x];

    } else {
        y_t = INT32_MAX;
    }

    __syncthreads();

    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-2)*(blockSize-2)];
    __shared__ int chunk_start[(blockSize-2)*(blockSize-2)];

    __syncthreads();
    if((threadIdx.z == 1) && (threadIdx.y == 1) && (threadIdx.x < (blockSize-2)*(blockSize-2))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-2;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();

    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 1 + (threadIdx.z - 1)*(blockSize-2);
    for(int j = 1; j < ((blockSize-2)*(blockSize-2)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    int sparse_block = 0;
    int sparse_block_t = 0;
    int sparse_block_p = 0;
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();
        while( y_0 < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block++;

            if( (sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        __syncwarp();
        while( y_t < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_t++;

            if( (sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row] ) {
                y_t = y_vec_tree[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
                f_t = tree_data[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
            } else {
                y_t = INT32_MAX;
            }
        }

        __syncwarp();
        while( y_p < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if( y_0 <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % N] = f_0;
        }
        __syncwarp();
        if( y_t <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+1) % N] = f_t;
        }
        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 1) % N] = f_p;
        }

        __syncthreads();
        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {
            float neigh_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0+1, neigh_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    } // end for y_chunk
}

/// -------------------------------------------------------- ///
///            with reflective boundary condition            ///
/// -------------------------------------------------------- ///

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_reflective(const uint64_t* __restrict__ level_xz_vec,
                                        const uint64_t* __restrict__ xz_end_vec,
                                        const uint16_t* __restrict__ y_vec,
                                        const inputType* __restrict__ input_particles,
                                        outputType* __restrict__ output_particles,
                                        const stencilType* __restrict__ stencil,
                                        const int z_num,
                                        const int x_num,
                                        const int y_num,
                                        const int z_num_parent,
                                        const int x_num_parent,
                                        const int level,
                                        const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    int x_index = index % x_num + threadIdx.y - 1;
    int z_index = index / x_num + threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (blockSize - 1)) &&
                     (threadIdx.z > 0) && (threadIdx.z < (blockSize - 1));

    if(x_index < 0) {
        x_index = -x_index;
        not_ghost = false;
    }

    if(x_index >= x_num) {
        x_index = x_num - 1 - (x_index - x_num + 1);
        not_ghost = false;
    }

    if(z_index < 0) {
        z_index = -z_index;
        not_ghost = false;
    }

    if(z_index >= z_num) {
        z_index = z_num - 1 - (z_index - z_num + 1);
        not_ghost = false;
    }

    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    const int x_index_p = x_index / 2;
    const int z_index_p = z_index / 2;

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }
    __syncthreads();

    if(global_index_begin_0_s[5] == global_index_end_0_s[5]) {
        return;
    }

    if(threadIdx.x == 0) {
        size_t xz_start = x_index_p + z_index_p * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    __syncthreads();

    inputType f_0, f_p;
    int y_0, y_p;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;

    if(update_index < global_index_end_0_s[row]) {
        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];
    } else {
        y_0 = INT32_MAX;
    }

    __syncthreads();

    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    // overlapping y chunks

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-2)*(blockSize-2)];
    __shared__ int chunk_start[(blockSize-2)*(blockSize-2)];

    __syncthreads();

    if((threadIdx.z == 1) && (threadIdx.y == 1) && (threadIdx.x < (blockSize-2)*(blockSize-2))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-2;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 1 + (threadIdx.z - 1)*(blockSize-2);
    for(int j = 1; j < ((blockSize-2)*(blockSize-2)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    int sparse_block = 0;
    int sparse_block_p = 0;
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();

        // update apr particle
        while( y_0 < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block++;

            if( (sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }
        __syncwarp();

        // update parent particle
        while( y_p < (y_chunk*chunkSizeInternal - 1)) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if(y_0 <= (y_chunk+1)*chunkSizeInternal) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % chunkSize] = f_0;
        }

        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num)) {
            local_patch[threadIdx.z][threadIdx.y][(y_p+1) % chunkSize] = f_p;
        }

        __syncthreads();
        if(y_chunk == 0) {
            if(threadIdx.x == 0) {
                local_patch[threadIdx.z][threadIdx.y][0] = local_patch[threadIdx.z][threadIdx.y][2];
            }
        }

        if( ((y_chunk+1)*chunkSizeInternal-1) > (y_num-2) ) {
            if(threadIdx.x == 0) {
                local_patch[threadIdx.z][threadIdx.y][(y_num+1) % N] = local_patch[threadIdx.z][threadIdx.y][(y_num-1) % N];
            }
        }

        __syncthreads();
        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 1, neighbour_sum)

        }
    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_reflective(const uint64_t* __restrict__ level_xz_vec,
                                             const uint64_t* __restrict__ xz_end_vec,
                                             const uint16_t* __restrict__ y_vec,
                                             const inputType* __restrict__ input_particles,
                                             outputType* __restrict__ output_particles,
                                             const stencilType* __restrict__ stencil,
                                             const uint64_t* __restrict__ level_xz_vec_tree,
                                             const uint64_t* __restrict__ xz_end_vec_tree,
                                             const uint16_t* __restrict__ y_vec_tree,
                                             const treeType* __restrict__ tree_data,
                                             const int z_num,
                                             const int x_num,
                                             const int y_num,
                                             const int z_num_parent,
                                             const int x_num_parent,
                                             const int level,
                                             const int* __restrict__ offset_ind) {

    const int index = offset_ind[blockIdx.x];

    int x_index = index % x_num + (int)threadIdx.y - 1;
    int z_index = index / x_num + (int)threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (blockSize - 1)) &&
                     (threadIdx.z > 0) && (threadIdx.z < (blockSize - 1));

    if(x_index < 0) {
        x_index = -x_index;
        not_ghost = false;
    }

    if(x_index >= x_num) {
        x_index = x_num - 1 - (x_index - x_num + 1);
        not_ghost = false;
    }

    if(z_index < 0) {
        z_index = -z_index;
        not_ghost = false;
    }

    if(z_index >= z_num) {
        z_index = z_num - 1 - (z_index - z_num + 1);
        not_ghost = false;
    }

    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_t_s[blockSize*blockSize];
    __shared__ size_t global_index_end_t_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    const int x_index_p = x_index / 2;
    const int z_index_p = z_index / 2;

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 1) {
        size_t xz_start = x_index_p + z_index_p * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 2) {
        size_t xz_start = level_xz_vec_tree[level] + (x_index) + (z_index) * x_num;
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];
    }

    __syncthreads();

    stencilType f_0, f_p, f_t;
    int y_0, y_p, y_t;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;

    if((update_index) < global_index_end_0_s[row]) {

        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];

    } else {
        y_0 = INT32_MAX;
    }

    __syncthreads();

    if((global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row]) {

        f_t = tree_data[global_index_begin_t_s[row] + threadIdx.x];
        y_t = y_vec_tree[global_index_begin_t_s[row] + threadIdx.x];

    } else {
        y_t = INT32_MAX;
    }

    __syncthreads();

    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-2)*(blockSize-2)];
    __shared__ int chunk_start[(blockSize-2)*(blockSize-2)];

    __syncthreads();
    if((threadIdx.z == 1) && (threadIdx.y == 1) && (threadIdx.x < (blockSize-2)*(blockSize-2))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-2;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();

    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 1 + (threadIdx.z - 1)*(blockSize-2);
    for(int j = 1; j < ((blockSize-2)*(blockSize-2)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    int sparse_block = 0;
    int sparse_block_t = 0;
    int sparse_block_p = 0;
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();
        while( y_0 < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block++;

            if( (sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        __syncwarp();
        while( y_t < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_t++;

            if( (sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row] ) {
                y_t = y_vec_tree[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
                f_t = tree_data[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
            } else {
                y_t = INT32_MAX;
            }
        }

        __syncwarp();
        while( y_p < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if( y_0 <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % N] = f_0;
        }

        __syncwarp();
        if( y_t <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+1) % N] = f_t;
        }

        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 1) % N] = f_p;
        }

        __syncthreads();
        if(y_chunk == 0) {
            if(threadIdx.x == 0) {
                local_patch[threadIdx.z][threadIdx.y][0] = local_patch[threadIdx.z][threadIdx.y][2];
            }
        }

        if( ((y_chunk+1)*chunkSizeInternal) > (y_num-1) ) {
            if(threadIdx.x == 0) {
                local_patch[threadIdx.z][threadIdx.y][(y_num+1) % N] = local_patch[threadIdx.z][threadIdx.y][(y_num-1) % N];
            }
        }

        __syncthreads();
        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {
            float neigh_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0+1, neigh_sum)
        }
    } // end for y_chunk
}


/// -------------------------------------------------------- ///
///           without non-empty rows precomputation          ///
/// -------------------------------------------------------- ///

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_chunked(const uint64_t* __restrict__ level_xz_vec,
                                     const uint64_t* __restrict__ xz_end_vec,
                                     const uint16_t* __restrict__ y_vec,
                                     const inputType* __restrict__ input_particles,
                                     outputType* __restrict__ output_particles,
                                     const stencilType* __restrict__ stencil,
                                     const int z_num,
                                     const int x_num,
                                     const int y_num,
                                     const int z_num_parent,
                                     const int x_num_parent,
                                     const int level) {

    const int z_index = blockIdx.z * (blockSize-2) + threadIdx.z - 1;
    const int x_index = blockIdx.x * (blockSize-2) + threadIdx.y - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {
        return; // out of bounds
    }

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (blockSize - 1)) &&
                           (threadIdx.z > 0) && (threadIdx.z < (blockSize - 1));

    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    const int x_index_p = x_index / 2;
    const int z_index_p = z_index / 2;

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }
    __syncthreads();

    if(global_index_begin_0_s[5] == global_index_end_0_s[5]) {
        return;
    }

    if(threadIdx.x == 0) {
        size_t xz_start = x_index_p + z_index_p * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    __syncthreads();

    stencilType f_0, f_p;
    int y_0, y_p;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;

    if((update_index) < global_index_end_0_s[row]) {
        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];
    } else {
        y_0 = INT32_MAX;
    }

    __syncthreads();
    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    // overlapping y chunks

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-2)*(blockSize-2)];
    __shared__ int chunk_start[(blockSize-2)*(blockSize-2)];

    __syncthreads();

    if((threadIdx.z == 1) && (threadIdx.y == 1) && (threadIdx.x < (blockSize-2)*(blockSize-2))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-2;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }
    __syncthreads();

    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 1 + (threadIdx.z - 1)*(blockSize-2);
    for(int j = 1; j < ((blockSize-2)*(blockSize-2)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    int sparse_block = 0;
    int sparse_block_p = 0;
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();

        while( y_0 < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block++;
            if( (sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        __syncwarp();
        while( y_p < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if( y_0 <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % chunkSize] = f_0;
        }

        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p+1) % chunkSize] = f_p;
        }

        __syncthreads();
        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {
            float neighbour_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 1, neighbour_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_chunked(const uint64_t* __restrict__ level_xz_vec,
                                          const uint64_t* __restrict__ xz_end_vec,
                                          const uint16_t* __restrict__ y_vec,
                                          const inputType* __restrict__ input_particles,
                                          outputType* __restrict__ output_particles,
                                          const stencilType* __restrict__ stencil,
                                          const uint64_t* __restrict__ level_xz_vec_tree,
                                          const uint64_t* __restrict__ xz_end_vec_tree,
                                          const uint16_t* __restrict__ y_vec_tree,
                                          const treeType* __restrict__ tree_data,
                                          const int z_num,
                                          const int x_num,
                                          const int y_num,
                                          const int z_num_parent,
                                          const int x_num_parent,
                                          const int level) {

    const int z_index = blockIdx.z * (blockSize-2) + threadIdx.z - 1;
    const int x_index = blockIdx.x * (blockSize-2) + threadIdx.y - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {
        return; // out of bounds
    }

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (blockSize - 1)) &&
                           (threadIdx.z > 0) && (threadIdx.z < (blockSize - 1));

    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_t_s[blockSize*blockSize];
    __shared__ size_t global_index_end_t_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    const int x_index_p = x_index / 2;
    const int z_index_p = z_index / 2;

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 1) {
        size_t xz_start = x_index_p + z_index_p * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 2) {
        size_t xz_start = level_xz_vec_tree[level] + (x_index) + (z_index) * x_num;
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];
    }

    __syncthreads();

    stencilType f_0, f_p, f_t;
    int y_0, y_p, y_t;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;

    if((update_index) < global_index_end_0_s[row]) {

        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];

    } else {
        y_0 = INT32_MAX;
    }

    __syncthreads();

    if((global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row]) {

        f_t = tree_data[global_index_begin_t_s[row] + threadIdx.x];
        y_t = y_vec_tree[global_index_begin_t_s[row] + threadIdx.x];

    } else {
        y_t = INT32_MAX;
    }

    __syncthreads();

    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-2)*(blockSize-2)];
    __shared__ int chunk_start[(blockSize-2)*(blockSize-2)];

    __syncthreads();
    if((threadIdx.z == 1) && (threadIdx.y == 1) && (threadIdx.x < (blockSize-2)*(blockSize-2))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-2;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-1 + (threadIdx.z-1)*(blockSize-2)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 1 + (threadIdx.z - 1)*(blockSize-2);
    for(int j = 1; j < ((blockSize-2)*(blockSize-2)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    int sparse_block = 0;
    int sparse_block_t = 0;
    int sparse_block_p = 0;
    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();
        while( (y_0 < (y_chunk*chunkSizeInternal - 1)) ) {
            sparse_block++;

            if( (sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*chunkSize + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        __syncwarp();
        while( y_t < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_t++;

            if( (sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row] ) {
                y_t = y_vec_tree[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
                f_t = tree_data[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
            } else {
                y_t = INT32_MAX;
            }
        }

        __syncwarp();
        while( y_p < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if( y_0 <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % N] = f_0;
        }

        __syncwarp();
        if( y_t <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+1) % N] = f_t;
        }

        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 1) % N] = f_p;
        }

        __syncthreads();
        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {
            float neigh_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0+1, neigh_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    } // end for y_chunk
}


/// -------------------------------------------------------- ///
///          helper functions to launch the kernels          ///
/// -------------------------------------------------------- ///


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter, bool use_stencil_downsample) {

    const int blockSize = 4;
    const int chunkSize = 32;
    size_t stencil_offset = 0;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        int ne_sz = ne_counter[level+1] - ne_counter[level];
        int offset = ne_counter[level];

        if( ne_sz == 0) {
            if(use_stencil_downsample) {
                stencil_offset += 27;
            }
            continue;
        }

        dim3 blocks_l(ne_sz, 1, 1);
        dim3 threads_l(chunkSize, blockSize, blockSize);

        if (level == access.level_max()) {
            conv_max_333_chunked
                    <chunkSize, blockSize>
                    << < blocks_l, threads_l >> >
                                   ( access.get_level_xz_vec_ptr(),
                                           access.get_xz_end_vec_ptr(),
                                           access.get_y_vec_ptr(),
                                           input_gpu,
                                           output_gpu,
                                           stencil_gpu,
                                           access.z_num(level),
                                           access.x_num(level),
                                           access.y_num(level),
                                           tree_access.z_num(level-1),
                                           tree_access.x_num(level-1),
                                           level,
                                           ne_rows_gpu + offset);

        } else {
            conv_interior_333_chunked
                    <chunkSize, blockSize>
                    <<< blocks_l, threads_l >>>
                                  (access.get_level_xz_vec_ptr(),
                                          access.get_xz_end_vec_ptr(),
                                          access.get_y_vec_ptr(),
                                          input_gpu,
                                          output_gpu,
                                          stencil_gpu + stencil_offset,
                                          tree_access.get_level_xz_vec_ptr(),
                                          tree_access.get_xz_end_vec_ptr(),
                                          tree_access.get_y_vec_ptr(),
                                          tree_data_gpu,
                                          access.z_num(level),
                                          access.x_num(level),
                                          access.y_num(level),
                                          tree_access.z_num(level - 1),
                                          tree_access.x_num(level - 1),
                                          level,
                                          ne_rows_gpu + offset);
        }

        if(use_stencil_downsample) {
            stencil_offset += 27;
        }
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333_reflective(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                                       stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter, bool use_stencil_downsample){

    const int blockSize = 4;
    const int chunkSize = 32;
    int stencil_offset = 0;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        int ne_sz = ne_counter[level+1] - ne_counter[level];
        int offset = ne_counter[level];

        if( ne_sz == 0) {
            if(use_stencil_downsample) {
                stencil_offset += 27;
            }
            continue;
        }

        dim3 blocks_l(ne_sz, 1, 1);
        dim3 threads_l(chunkSize, blockSize, blockSize);

        if (level == access.level_max()) {
            conv_max_333_reflective
                    <chunkSize, blockSize>
                    << < blocks_l, threads_l >> >
                                   ( access.get_level_xz_vec_ptr(),
                                           access.get_xz_end_vec_ptr(),
                                           access.get_y_vec_ptr(),
                                           input_gpu,
                                           output_gpu,
                                           stencil_gpu,
                                           access.z_num(level),
                                           access.x_num(level),
                                           access.y_num(level),
                                           tree_access.z_num(level-1),
                                           tree_access.x_num(level-1),
                                           level,
                                           ne_rows_gpu + offset);

        } else {
            conv_interior_333_reflective
                    <chunkSize, blockSize>
                    <<< blocks_l, threads_l >>>
                                  (access.get_level_xz_vec_ptr(),
                                          access.get_xz_end_vec_ptr(),
                                          access.get_y_vec_ptr(),
                                          input_gpu,
                                          output_gpu,
                                          stencil_gpu + stencil_offset,
                                          tree_access.get_level_xz_vec_ptr(),
                                          tree_access.get_xz_end_vec_ptr(),
                                          tree_access.get_y_vec_ptr(),
                                          tree_data_gpu,
                                          access.z_num(level),
                                          access.x_num(level),
                                          access.y_num(level),
                                          tree_access.z_num(level - 1),
                                          tree_access.x_num(level - 1),
                                          level,
                                          ne_rows_gpu + offset);
        }

        if(use_stencil_downsample) {
            stencil_offset += 27;
        }
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                                       stencilType* stencil_gpu, treeType* tree_data_gpu, bool use_stencil_downsample) {

    const int blockSize = 4;
    const int chunkSize = 32;
    size_t stencil_offset = 0;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        const int blocks_x = (access.x_num(level) + blockSize - 3) / (blockSize - 2);
        const int blocks_z = (access.z_num(level) + blockSize - 3) / (blockSize - 2);

        dim3 blocks_l(blocks_x, 1, blocks_z);
        dim3 threads_l(chunkSize, blockSize, blockSize);

        if (level == access.level_max()) {

            conv_max_333_chunked
                    <chunkSize, blockSize>
                    << < blocks_l, threads_l >> >
                                   ( access.get_level_xz_vec_ptr(),
                                           access.get_xz_end_vec_ptr(),
                                           access.get_y_vec_ptr(),
                                           input_gpu,
                                           output_gpu,
                                           stencil_gpu,
                                           access.z_num(level),
                                           access.x_num(level),
                                           access.y_num(level),
                                           tree_access.z_num(level-1),
                                           tree_access.x_num(level-1),
                                           level);

        } else {

            conv_interior_333_chunked
                    <chunkSize, blockSize>
                    <<< blocks_l, threads_l >>>
                                  (access.get_level_xz_vec_ptr(),
                                          access.get_xz_end_vec_ptr(),
                                          access.get_y_vec_ptr(),
                                          input_gpu,
                                          output_gpu,
                                          stencil_gpu + stencil_offset,
                                          tree_access.get_level_xz_vec_ptr(),
                                          tree_access.get_xz_end_vec_ptr(),
                                          tree_access.get_y_vec_ptr(),
                                          tree_data_gpu,
                                          access.z_num(level),
                                          access.x_num(level),
                                          access.y_num(level),
                                          tree_access.z_num(level - 1),
                                          tree_access.x_num(level - 1),
                                          level);
        }
        if(use_stencil_downsample) {
            stencil_offset += 27;
        }
    }
}


/// -------------------------------------------------------- ///
///            functions including data transfers            ///
/// -------------------------------------------------------- ///


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                            VectorData<stencilType>& stencil, VectorData<treeType>& tree_data, bool reflective_bc, bool use_stencil_downsample, bool normalize_stencil) {
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel
     *  conv_stencil needs to have 27 entries, with element (x, y, z) corresponding to index z*9 + x*3 + y
     */

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 27);

    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());

    /// compute nonempty rows
    VectorData<int> ne_counter_ds; //non empty rows
    VectorData<int> ne_counter;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;

    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
    compute_ne_rows_cuda<16, 32>(access, ne_counter, ne_rows_gpu, 2);

    /// downsample the stencil
    VectorData<stencilType> stencil_vec;
    if(use_stencil_downsample) {
        APRStencil::get_downsampled_stencils(stencil, stencil_vec, access.level_max() - access.level_min(), normalize_stencil);
    }

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu;

    if(use_stencil_downsample) {
        stencil_gpu.initialize(stencil_vec.data(), stencil_vec.size());
    } else {
        stencil_gpu.initialize(stencil.data(), stencil.size());
    }

    /// copy input particles and stencil(s) to device
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    /// Fill the APR Tree by average downsampling
    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    error_check( cudaDeviceSynchronize() )

    /// perform the convolution operation
    if(reflective_bc) {
        isotropic_convolve_333_reflective(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                                          tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, use_stencil_downsample);
    } else {
        isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                               tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, use_stencil_downsample);
    }
    error_check( cudaDeviceSynchronize() )

    /// transfer the results back to the host
    output_gpu.copyD2H();
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                                VectorData<stencilType>& stencil, VectorData<treeType>& tree_data, bool use_stencil_downsample, bool normalize_stencil){
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel
     *  conv_stencil needs to have 27 entries, with element (x, y, z) corresponding to index z*9 + x*3 + y
     */

    /// transfer APR access structure to the GPU
    tree_access.init_gpu();
    access.init_gpu(tree_access);

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 27);

    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());

    /// downsample the stencil
    VectorData<stencilType> stencil_vec;
    if(use_stencil_downsample) {
        APRStencil::get_downsampled_stencils(stencil, stencil_vec, access.level_max() - access.level_min(), normalize_stencil);
    }

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu;

    if(use_stencil_downsample) {
        stencil_gpu.initialize(stencil_vec.data(), stencil_vec.size());
    } else {
        stencil_gpu.initialize(stencil.data(), stencil.size());
    }

    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    /// fill the APR Tree by average downsampling
    downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    error_check( cudaDeviceSynchronize() )

    /// run convolution kernels
    isotropic_convolve_333_alt(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), use_stencil_downsample);
    error_check( cudaDeviceSynchronize() )

    /// transfer the results back to the host
    output_gpu.copyD2H();
}


/// instantiate templates
template void isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, bool, bool, bool);
template void isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, bool, bool, bool);
template void isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<double>&, bool, bool, bool);

template void isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, bool, bool);
template void isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, bool, bool);
template void isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<double>&, bool, bool);

