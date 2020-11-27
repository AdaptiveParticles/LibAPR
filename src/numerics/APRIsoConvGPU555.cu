//
// Created by joel on 08.04.20.
//

#include "APRIsoConvGPU555.hpp"



template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
__launch_bounds__(1024, 1)
conv_max_555_chunked(const uint64_t* __restrict__ level_xz_vec,
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

    const int x_index = index % x_num + threadIdx.y - 2;
    const int z_index = index / x_num + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[5][5][5];

    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }
    __syncwarp();

    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {
        return; // out of bounds
    }

    const bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                           (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

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

    if(threadIdx.x == 1) {
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

    __syncwarp();
    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    int sparse_block = 0;
    int sparse_block_p = 0;

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-4)*(blockSize-4)];
    __shared__ int chunk_start[(blockSize-4)*(blockSize-4)];

    __syncthreads();

    if((threadIdx.z == 2) && (threadIdx.y == 2) && (threadIdx.x < (blockSize-4)*(blockSize-4))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-4;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 2 + (threadIdx.z - 2)*(blockSize-4);
    for(int j = 1; j < ((blockSize-4)*(blockSize-4)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();
        while( y_0 < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block++;
            if( (sparse_block*N + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {
                update_index = sparse_block*N + global_index_begin_0_s[row] + threadIdx.x;
                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        __syncwarp();
        while( (y_p < (y_chunk*chunkSizeInternal - 2)) ) {
            sparse_block_p++;
            if( (global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0 + 2) % N] = f_0;
        }

        __syncwarp();
        if( (y_p <= ((y_chunk+1)*chunkSizeInternal+1)) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 2) % N] = f_p;
        }

        __syncthreads();

        if( ((y_0 >= (y_chunk*chunkSizeInternal)) && (y_0 < ((y_chunk+1)*chunkSizeInternal))) ) {
            float neighbour_sum = 0;
            LOCALPATCHCONV555_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 2, neighbour_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void
__launch_bounds__(1024, 1)
conv_interior_555_chunked(const uint64_t* __restrict__ level_xz_vec,
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

    const int x_index = index % x_num + threadIdx.y - 2;
    const int z_index = index / x_num + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    /// copy the stencil to shared memory
    __shared__ stencilType local_stencil[5][5][5];
    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }
    __syncwarp();

    /// initialize "local isotropic patch" buffer in shared memory
    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0; // zero pad

    /// if out of bounds, return
    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {
        return;
    }

    /// the 2 outer rows/columns are "ghosts" -- the output is only computed at non-ghost locations.
    const bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                           (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

    /// the begin and end indices of each x-z pair are held in shared memory
    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_t_s[blockSize*blockSize];
    __shared__ size_t global_index_end_t_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    __syncthreads();

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 1) {
        size_t xz_start = (x_index / 2) + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 2) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];
    }

    __syncthreads();

    /// each thread grabs a particle in its designated row (x-z pair) from the current APR level (y_0, f_0), the
    /// current level in the APRTree (y_t, f_t) and the parent APR level (y_p, f_p)
    stencilType f_0, f_t, f_p;
    int y_0, y_t, y_p;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;
    if( (update_index < global_index_end_0_s[row]) ) {
        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];
    } else {
        y_0 = INT32_MAX/2;
    }

    __syncwarp();

    if( ((global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row]) ) {
        f_t = tree_data[global_index_begin_t_s[row] + threadIdx.x];
        y_t = y_vec_tree[global_index_begin_t_s[row] + threadIdx.x];
    } else {
        y_t = INT32_MAX;
    }

    __syncwarp();

    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    /// The y-dimension will be looped over in "chunks" - we first determine the range of y-values that must be included.
    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-4)*(blockSize-4)];
    __shared__ int chunk_start[(blockSize-4)*(blockSize-4)];

    __syncthreads();
    if((threadIdx.z == 2) && (threadIdx.y == 2) && (threadIdx.x < (blockSize-4)*(blockSize-4))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-4;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 2 + (threadIdx.z - 2)*(blockSize-4);
    for(int j = 1; j < ((blockSize-4)*(blockSize-4)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    __syncthreads();

    int sparse_block = 0;
    int sparse_block_p = 0;
    int sparse_block_t = 0;

    /// Loop over the y-dimension in chunks. Each thread holds a particle; when that particle is within the chunk, it is
    /// inserted into the local patch. If the chunk has passed the y coordinate of the current particle, the thread grabs
    /// a new one. This is done similarly for all 3 particle types (APR, tree and parent).
    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        // update APR particle
        __syncthreads();
        while( y_0 < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block++;
            if( (sparse_block*N + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*N + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        // update tree particle
        __syncwarp();
        while( y_t < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block_t++;
            if( (sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row] ) {
                y_t = y_vec_tree[sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x];
                f_t = tree_data[sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x];
            } else {
                y_t = INT32_MAX;
            }
        }

        // update parent particle
        __syncwarp();
        while( (y_p < (y_chunk*chunkSizeInternal - 2)) ) {
            sparse_block_p++;
            if( (global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + sparse_block_p*(N/2) + threadIdx.x/2] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + sparse_block_p*(N/2) + threadIdx.x/2];
            } else{
                y_p = INT32_MAX;
            }
        }

        // insert APR particle
        __syncwarp();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+2) % N] = f_0;
        }

        // insert tree particle
        __syncwarp();
        if( y_t <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+2) % N] = f_t;
        }

        // insert parent particle
        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal+1) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p+2) % N] = f_p;
        }

        // compute output value
        __syncthreads();
        if( (y_0 >= (y_chunk*chunkSizeInternal)) && (y_0 < ((y_chunk+1)*chunkSizeInternal)) ) {
            float neigh_sum;
            LOCALPATCHCONV555_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 2, neigh_sum)
        }

        // reset local patch
        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    } // end for y_chunk
}



template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
__launch_bounds__(1024, 1)
conv_max_555_reflective(const uint64_t* __restrict__ level_xz_vec,
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

    const unsigned int N = chunkSize;
    const int index = offset_ind[blockIdx.x];

    int x_index = index % x_num + (int)threadIdx.y - 2;
    int z_index = index / x_num + (int)threadIdx.z - 2;

    __shared__ stencilType local_stencil[5][5][5];

    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }
    __syncwarp();

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                     (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

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

    if(threadIdx.x == 1) {
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

    __syncwarp();
    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    int sparse_block = 0;
    int sparse_block_p = 0;

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-4)*(blockSize-4)];
    __shared__ int chunk_start[(blockSize-4)*(blockSize-4)];

    __syncthreads();

    if((threadIdx.z == 2) && (threadIdx.y == 2) && (threadIdx.x < (blockSize-4)*(blockSize-4))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-4;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 2 + (threadIdx.z - 2)*(blockSize-4);
    for(int j = 1; j < ((blockSize-4)*(blockSize-4)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();
        while( y_0 < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block++;
            if( (sparse_block*N + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {
                update_index = sparse_block*N + global_index_begin_0_s[row] + threadIdx.x;
                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        __syncwarp();
        while( (y_p < (y_chunk*chunkSizeInternal - 2)) ) {
            sparse_block_p++;
            if( (global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0 + 2) % N] = f_0;
        }

        __syncwarp();
        if( (y_p <= ((y_chunk+1)*chunkSizeInternal+1)) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 2) % N] = f_p;
        }

        __syncthreads();
        if(y_chunk == 0) {
            if(threadIdx.x < 2) {
                local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = local_patch[threadIdx.z][threadIdx.y][4-threadIdx.x];
            }
        }

        if( ((y_chunk+1)*chunkSizeInternal) > (y_num-2) ) {
            int limit = max( (y_chunk+1)*chunkSizeInternal - (y_num - 2), 2 );
            if(threadIdx.x < limit) {
                local_patch[threadIdx.z][threadIdx.y][(y_num+threadIdx.x+2) % N] = local_patch[threadIdx.z][threadIdx.y][(y_num-threadIdx.x) % N];
            }
        }

        __syncthreads();

        if( ((y_0 >= (y_chunk*chunkSizeInternal)) && (y_0 < ((y_chunk+1)*chunkSizeInternal))) ) {
            float neighbour_sum = 0;
            LOCALPATCHCONV555_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 2, neighbour_sum)
        }
    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void
__launch_bounds__(1024, 1)
conv_interior_555_reflective(const uint64_t* __restrict__ level_xz_vec,
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

    int x_index = index % x_num + (int)threadIdx.y - 2;
    int z_index = index / x_num + (int)threadIdx.z - 2;

    const unsigned int N = chunkSize;

    /// copy the stencil to shared memory
    __shared__ stencilType local_stencil[5][5][5];
    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }
    __syncwarp();

    /// initialize "local isotropic patch" buffer in shared memory
    __shared__ stencilType local_patch[blockSize][blockSize][N];

    bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                     (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

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

    /// the begin and end indices of each x-z pair are held in shared memory
    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_t_s[blockSize*blockSize];
    __shared__ size_t global_index_end_t_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    __syncthreads();

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 1) {
        size_t xz_start = (x_index / 2) + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 2) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];
    }

    __syncthreads();

    /// each thread grabs a particle in its designated row (x-z pair) from the current APR level (y_0, f_0), the
    /// current level in the APRTree (y_t, f_t) and the parent APR level (y_p, f_p)
    stencilType f_0, f_t, f_p;
    int y_0, y_t, y_p;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;
    if( (update_index < global_index_end_0_s[row]) ) {
        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];
    } else {
        y_0 = INT32_MAX/2;
    }

    __syncwarp();

    if( ((global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row]) ) {
        f_t = tree_data[global_index_begin_t_s[row] + threadIdx.x];
        y_t = y_vec_tree[global_index_begin_t_s[row] + threadIdx.x];
    } else {
        y_t = INT32_MAX;
    }

    __syncwarp();

    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    /// The y-dimension will be looped over in "chunks" - we first determine the range of y-values that must be included.
    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-4)*(blockSize-4)];
    __shared__ int chunk_start[(blockSize-4)*(blockSize-4)];

    __syncthreads();
    if((threadIdx.z == 2) && (threadIdx.y == 2) && (threadIdx.x < (blockSize-4)*(blockSize-4))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-4;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 2 + (threadIdx.z - 2)*(blockSize-4);
    for(int j = 1; j < ((blockSize-4)*(blockSize-4)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    __syncthreads();

    int sparse_block = 0;
    int sparse_block_p = 0;
    int sparse_block_t = 0;

    /// Loop over the y-dimension in chunks. Each thread holds a particle; when that particle is within the chunk, it is
    /// inserted into the local patch. If the chunk has passed the y coordinate of the current particle, the thread grabs
    /// a new one. This is done similarly for all 3 particle types (APR, tree and parent).
    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        // update APR particle
        __syncthreads();
        while( y_0 < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block++;
            if( (sparse_block*N + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*N + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        // update tree particle
        __syncwarp();
        while( y_t < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block_t++;
            if( (sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row] ) {
                y_t = y_vec_tree[sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x];
                f_t = tree_data[sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x];
            } else {
                y_t = INT32_MAX;
            }
        }

        // update parent particle
        __syncwarp();
        while( (y_p < (y_chunk*chunkSizeInternal - 2)) ) {
            sparse_block_p++;
            if( (global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + sparse_block_p*(N/2) + threadIdx.x/2] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + sparse_block_p*(N/2) + threadIdx.x/2];
            } else{
                y_p = INT32_MAX;
            }
        }

        // insert APR particle
        __syncwarp();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+2) % N] = f_0;
        }

        // insert tree particle
        __syncwarp();
        if( y_t <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+2) % N] = f_t;
        }

        // insert parent particle
        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal+1) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p+2) % N] = f_p;
        }

        __syncthreads();
        if(y_chunk == 0) {
            if(threadIdx.x < 2) {
                local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = local_patch[threadIdx.z][threadIdx.y][4-threadIdx.x];
            }
        }

        if( ((y_chunk+1)*chunkSizeInternal) > (y_num-2) ) {
            int limit = max( (y_chunk+1)*chunkSizeInternal - (y_num - 2), 2 );
            if(threadIdx.x < limit) {
                local_patch[threadIdx.z][threadIdx.y][(y_num+threadIdx.x+2) % N] = local_patch[threadIdx.z][threadIdx.y][(y_num-threadIdx.x) % N];
            }
        }

        // compute output value
        __syncthreads();
        if( (y_0 >= (y_chunk*chunkSizeInternal)) && (y_0 < ((y_chunk+1)*chunkSizeInternal)) ) {
            float neigh_sum;
            LOCALPATCHCONV555_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 2, neigh_sum)
        }
    } // end for y_chunk
}



template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
__launch_bounds__(1024, 1)
conv_max_555_chunked(const uint64_t* __restrict__ level_xz_vec,
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

    const int x_index = blockIdx.x * (blockSize - 4) + threadIdx.y - 2;
    const int z_index = blockIdx.z * (blockSize - 4) + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[5][5][5];

    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }
    __syncwarp();

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {

        // out of bounds --> zero pad and return
        return;
    }

    const bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                           (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

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

    if(threadIdx.x == 1) {
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

    __syncwarp();
    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }


    int sparse_block = 0;
    int sparse_block_p = 0;

    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-4)*(blockSize-4)];
    __shared__ int chunk_start[(blockSize-4)*(blockSize-4)];

    __syncthreads();

    if((threadIdx.z == 2) && (threadIdx.y == 2) && (threadIdx.x < (blockSize-4)*(blockSize-4))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-4;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 2 + (threadIdx.z - 2)*(blockSize-4);
    for(int j = 1; j < ((blockSize-4)*(blockSize-4)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    __syncthreads();

    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        __syncthreads();
        while( y_0 < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block++;
            if( (sparse_block*N + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {
                update_index = sparse_block*N + global_index_begin_0_s[row] + threadIdx.x;
                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        __syncwarp();
        while( (y_p < (y_chunk*chunkSizeInternal - 2)) ) {
            sparse_block_p++;
            if( (global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncwarp();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0 + 2) % N] = f_0;
        }

        __syncwarp();
        if( (y_p <= ((y_chunk+1)*chunkSizeInternal+1)) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 2) % N] = f_p;
        }

        __syncthreads();

        if( ((y_0 >= (y_chunk*chunkSizeInternal)) && (y_0 < ((y_chunk+1)*chunkSizeInternal))) ) {
            float neighbour_sum = 0;
            LOCALPATCHCONV555_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 2, neighbour_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void
__launch_bounds__(1024, 1)
conv_interior_555_chunked(const uint64_t* __restrict__ level_xz_vec,
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

    const int x_index = blockIdx.x * (blockSize - 4) + threadIdx.y - 2;
    const int z_index = blockIdx.z * (blockSize - 4) + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    /// copy the stencil to shared memory
    __shared__ stencilType local_stencil[5][5][5];
    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }
    __syncwarp();

    /// initialize "local isotropic patch" buffer in shared memory
    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0; // zero pad

    /// if out of bounds, return
    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {
        return;
    }

    /// the 2 outer rows/columns are "ghosts" -- the output is only computed at non-ghost locations.
    const bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                           (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

    /// the begin and end indices of each x-z pair are held in shared memory
    const int row = threadIdx.y + threadIdx.z * blockSize;

    __shared__ size_t global_index_begin_0_s[blockSize*blockSize];
    __shared__ size_t global_index_end_0_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_t_s[blockSize*blockSize];
    __shared__ size_t global_index_end_t_s[blockSize*blockSize];

    __shared__ size_t global_index_begin_p_s[blockSize*blockSize];
    __shared__ size_t global_index_end_p_s[blockSize*blockSize];

    __syncthreads();

    if(threadIdx.x == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_0_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 1) {
        size_t xz_start = (x_index / 2) + (z_index / 2) * x_num_parent + level_xz_vec[level - 1];
        global_index_begin_p_s[row] = xz_end_vec[xz_start - 1];
        global_index_end_p_s[row] = xz_end_vec[xz_start];
    }

    if(threadIdx.x == 2) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t_s[row] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t_s[row] = xz_end_vec_tree[xz_start];
    }

    __syncthreads();

    /// each thread grabs a particle in its designated row (x-z pair) from the current APR level (y_0, f_0), the
    /// current level in the APRTree (y_t, f_t) and the parent APR level (y_p, f_p)
    stencilType f_0, f_t, f_p;
    int y_0, y_t, y_p;

    size_t update_index = global_index_begin_0_s[row] + threadIdx.x;
    if( (update_index < global_index_end_0_s[row]) ) {
        f_0 = input_particles[update_index];
        y_0 = y_vec[update_index];
    } else {
        y_0 = INT32_MAX/2;
    }

    __syncwarp();

    if( ((global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row]) ) {
        f_t = tree_data[global_index_begin_t_s[row] + threadIdx.x];
        y_t = y_vec_tree[global_index_begin_t_s[row] + threadIdx.x];
    } else {
        y_t = INT32_MAX;
    }

    __syncwarp();

    const int y_offset_p = threadIdx.x % 2;

    if((global_index_begin_p_s[row] + threadIdx.x/2) < global_index_end_p_s[row]) {
        f_p = input_particles[global_index_begin_p_s[row] + threadIdx.x/2];
        y_p = 2*y_vec[global_index_begin_p_s[row] + threadIdx.x/2] + y_offset_p;
    } else {
        y_p = INT32_MAX;
    }

    /// The y-dimension will be looped over in "chunks" - we first determine the range of y-values that must be included.
    __shared__ int chunkSizeInternal;
    __shared__ int chunk_end[(blockSize-4)*(blockSize-4)];
    __shared__ int chunk_start[(blockSize-4)*(blockSize-4)];

    __syncthreads();
    if((threadIdx.z == 2) && (threadIdx.y == 2) && (threadIdx.x < (blockSize-4)*(blockSize-4))) {
        chunk_end[threadIdx.x] = 0;
        chunk_start[threadIdx.x] = INT32_MAX;

        if(threadIdx.x == 0) {
            chunkSizeInternal = chunkSize-4;
        }
    }

    __syncthreads();

    // each non-ghost row determines its required y range
    if( ((threadIdx.x == 0) && not_ghost) ) {
        chunk_start[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_0/chunkSizeInternal;
        chunk_end[threadIdx.y-2 + (threadIdx.z-2)*(blockSize-4)] = y_vec[max(global_index_end_0_s[row], (size_t)1)-1]/chunkSizeInternal + 1;
    }

    __syncthreads();
    // reduce to find the minimal range spanning all of the required indices
    int i = threadIdx.y - 2 + (threadIdx.z - 2)*(blockSize-4);
    for(int j = 1; j < ((blockSize-4)*(blockSize-4)); j*=2) {

        if( ((threadIdx.x == 0) && not_ghost) ) {
            if( (i % (j*2)) == 0 ) {
                chunk_start[i] = min(chunk_start[i], chunk_start[i + j]);
                chunk_end[i] = max(chunk_end[i], chunk_end[i+j]);
            }
        }
        __syncthreads();
    }

    __syncthreads();

    int sparse_block = 0;
    int sparse_block_p = 0;
    int sparse_block_t = 0;

    /// Loop over the y-dimension in chunks. Each thread holds a particle; when that particle is within the chunk, it is
    /// inserted into the local patch. If the chunk has passed the y coordinate of the current particle, the thread grabs
    /// a new one. This is done similarly for all 3 particle types (APR, tree and parent).
    for(int y_chunk = chunk_start[0]; y_chunk < chunk_end[0]; ++y_chunk) {

        // update APR particle
        __syncthreads();
        while( y_0 < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block++;
            if( (sparse_block*N + global_index_begin_0_s[row] + threadIdx.x) < global_index_end_0_s[row] ) {

                update_index = sparse_block*N + global_index_begin_0_s[row] + threadIdx.x;

                y_0 = y_vec[update_index];
                f_0 = input_particles[update_index];
            } else {
                y_0 = INT32_MAX;
            }
        }

        // update tree particle
        __syncwarp();
        while( y_t < (y_chunk*chunkSizeInternal - 2) ) {
            sparse_block_t++;
            if( (sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row] ) {
                y_t = y_vec_tree[sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x];
                f_t = tree_data[sparse_block_t*N + global_index_begin_t_s[row] + threadIdx.x];
            } else {
                y_t = INT32_MAX;
            }
        }

        // update parent particle
        __syncwarp();
        while( (y_p < (y_chunk*chunkSizeInternal - 2)) ) {
            sparse_block_p++;
            if( (global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + sparse_block_p*(N/2) + threadIdx.x/2] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + sparse_block_p*(N/2) + threadIdx.x/2];
            } else{
                y_p = INT32_MAX;
            }
        }

        // insert APR particle
        __syncwarp();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+2) % N] = f_0;
        }

        // insert tree particle
        __syncwarp();
        if( y_t <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+2) % N] = f_t;
        }

        // insert parent particle
        __syncwarp();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal+1) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p+2) % N] = f_p;
        }

        // compute output value
        __syncthreads();
        if( (y_0 >= (y_chunk*chunkSizeInternal)) && (y_0 < ((y_chunk+1)*chunkSizeInternal)) ) {
            float neigh_sum;
            LOCALPATCHCONV555_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 2, neigh_sum)
        }

        // reset local patch
        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    } // end for y_chunk
}



template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter){

    const int blockSize = 8;
    const int chunkSize = 16;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        size_t ne_sz = ne_counter[level+1] - ne_counter[level];
        size_t offset = ne_counter[level];

        if( ne_sz == 0) {
            continue;
        }

        dim3 blocks_l(ne_sz, 1, 1);
        dim3 threads_l(chunkSize, blockSize, blockSize);

        if(level == access.level_max()) {

            conv_max_555_chunked
                    <chunkSize, blockSize>
                         << < blocks_l, threads_l >> >
                                  (access.get_level_xz_vec_ptr(),
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

            conv_interior_555_chunked
                    <chunkSize, blockSize>
                         <<< blocks_l, threads_l >>>
                                 (access.get_level_xz_vec_ptr(),
                                  access.get_xz_end_vec_ptr(),
                                  access.get_y_vec_ptr(),
                                  input_gpu,
                                  output_gpu,
                                  stencil_gpu,
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
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_reflective(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,outputType* output_gpu,
                                       stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter){

    const int blockSize = 8;
    const int chunkSize = 16;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        size_t ne_sz = ne_counter[level+1] - ne_counter[level];
        size_t offset = ne_counter[level];

        if( ne_sz == 0) {
            continue;
        }

        dim3 blocks_l(ne_sz, 1, 1);
        dim3 threads_l(chunkSize, blockSize, blockSize);

        if(level == access.level_max()) {

            conv_max_555_reflective
                    <chunkSize, blockSize>
                         << < blocks_l, threads_l >> >
                                (access.get_level_xz_vec_ptr(),
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

            conv_interior_555_reflective
                    <chunkSize, blockSize>
                         <<< blocks_l, threads_l >>>
                                 (access.get_level_xz_vec_ptr(),
                                  access.get_xz_end_vec_ptr(),
                                  access.get_y_vec_ptr(),
                                  input_gpu,
                                  output_gpu,
                                  stencil_gpu,
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
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                                outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu) {

    const int blockSize = 8;
    const int chunkSize = 16;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        const int x_blocks = (access.x_num(level) + blockSize - 5) / (blockSize - 4);
        const int z_blocks = (access.z_num(level) + blockSize - 5) / (blockSize - 4);

        dim3 blocks_l(x_blocks, 1, z_blocks);
        dim3 threads_l(chunkSize, blockSize, blockSize);

        if (level == access.level_max()) {

            conv_max_555_chunked
                    <chunkSize, blockSize>
                        << < blocks_l, threads_l >> >
                                  (access.get_level_xz_vec_ptr(),
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

            conv_interior_555_chunked
                    <chunkSize, blockSize>
                        <<< blocks_l, threads_l >>>
                                 (access.get_level_xz_vec_ptr(),
                                  access.get_xz_end_vec_ptr(),
                                  access.get_y_vec_ptr(),
                                  input_gpu,
                                  output_gpu,
                                  stencil_gpu,
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
    }
}


/// downsample stencil

template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                               outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu,
                               int* ne_rows_555, VectorData<int>& ne_counter_555, int* ne_rows_333, VectorData<int>& ne_counter_333){

    const int blockSize_555 = 8;
    const int chunkSize_555 = 16;
    const int blockSize_333 = 4;
    const int chunkSize_333 = 32;

    size_t stencil_offset = 0;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        if (level == access.level_max()) {

            size_t ne_sz = ne_counter_555[level+1] - ne_counter_555[level];
            size_t offset = ne_counter_555[level];

            if( ne_sz == 0) {
                stencil_offset += 125;
                continue;
            }

            dim3 blocks_l(ne_sz, 1, 1);
            dim3 threads_l(chunkSize_555, blockSize_555, blockSize_555);

            conv_max_555_chunked
                    <chunkSize_555, blockSize_555>
                            << < blocks_l, threads_l >> >
                                      (access.get_level_xz_vec_ptr(),
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
                                       ne_rows_555 + offset);

            stencil_offset += 125;

        } else {

            size_t ne_sz = ne_counter_333[level+1] - ne_counter_333[level];
            size_t offset = ne_counter_333[level];

            if( ne_sz == 0) {
                stencil_offset += 27;
                continue;
            }

            dim3 blocks_l(ne_sz, 1, 1);
            dim3 threads_l(chunkSize_333, blockSize_333, blockSize_333);

            conv_interior_333_chunked
                     <chunkSize_333, blockSize_333>
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
                                    ne_rows_333 + offset);

            stencil_offset += 27;
        }
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds_reflective(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                                          outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_555,
                                          VectorData<int>& ne_counter_555, int* ne_rows_333, VectorData<int>& ne_counter_333){

    const int blockSize_555 = 8;
    const int chunkSize_555 = 16;
    const int blockSize_333 = 4;
    const int chunkSize_333 = 32;

    size_t stencil_offset = 0;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        if (level == access.level_max()) {

            size_t ne_sz = ne_counter_555[level+1] - ne_counter_555[level];
            size_t offset = ne_counter_555[level];

            if( ne_sz == 0) {
                stencil_offset += 125;
                continue;
            }

            dim3 blocks_l(ne_sz, 1, 1);
            dim3 threads_l(chunkSize_555, blockSize_555, blockSize_555);

            conv_max_555_reflective
                    <chunkSize_555, blockSize_555>
                        << < blocks_l, threads_l >> >
                                  (access.get_level_xz_vec_ptr(),
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
                                   ne_rows_555 + offset);

            stencil_offset += 125;

        } else {

            size_t ne_sz = ne_counter_333[level+1] - ne_counter_333[level];
            size_t offset = ne_counter_333[level];

            if( ne_sz == 0) {
                stencil_offset += 27;
                continue;
            }

            dim3 blocks_l(ne_sz, 1, 1);
            dim3 threads_l(chunkSize_333, blockSize_333, blockSize_333);

            conv_interior_333_reflective
                    <chunkSize_333, blockSize_333>
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
                                  ne_rows_333 + offset);

            stencil_offset += 27;
        }
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                                    outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu) {

    const int blockSize_555 = 8;
    const int chunkSize_555 = 16;
    const int blockSize_333 = 4;
    const int chunkSize_333 = 32;

    size_t stencil_offset = 0;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        if (level == access.level_max()) {

            const int x_blocks = (access.x_num(level) + blockSize_555 - 5) / (blockSize_555 - 4);
            const int z_blocks = (access.z_num(level) + blockSize_555 - 5) / (blockSize_555 - 4);

            dim3 blocks_l(x_blocks, 1, z_blocks);
            dim3 threads_l(chunkSize_555, blockSize_555, blockSize_555);

            conv_max_555_chunked
                    <chunkSize_555, blockSize_555>
                        << < blocks_l, threads_l >> >
                                  (access.get_level_xz_vec_ptr(),
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

            stencil_offset += 125;

        } else {

            const int x_blocks = (access.x_num(level) + blockSize_333 - 3) / (blockSize_333 - 2);
            const int z_blocks = (access.z_num(level) + blockSize_333 - 3) / (blockSize_333 - 2);

            dim3 blocks_l(x_blocks, 1, z_blocks);
            dim3 threads_l(chunkSize_333, blockSize_333, blockSize_333);

            conv_interior_333_chunked
                    <chunkSize_333, blockSize_333>
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

            stencil_offset += 27;
        }
    }
}



template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                            VectorData<stencilType>& stencil, VectorData<treeType>& tree_data, bool reflective_bc, bool use_stencil_downsample,
                            bool normalize_stencil) {

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 125);

    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());

    VectorData<int> ne_counter_555;
    VectorData<int> ne_counter_333;
    VectorData<int> ne_counter_ds;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_555_gpu;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_333_gpu;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu;

    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
    compute_ne_rows_cuda<16, 32>(access, ne_counter_555, ne_rows_555_gpu, 4);

    VectorData<stencilType> stencil_vec;
    if(use_stencil_downsample) {
        compute_ne_rows_cuda<16, 32>(access, ne_counter_333, ne_rows_333_gpu, 2);

        APRStencil::get_downsampled_stencils(stencil, stencil_vec, access.level_max() - access.level_min(), normalize_stencil);
    }

    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu;

    if(use_stencil_downsample) {
        stencil_gpu.initialize(stencil_vec.data(), stencil_vec.size());
    } else {
        stencil_gpu.initialize(stencil.data(), stencil.size());
    }

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    /// Fill the APR Tree by average downsampling
    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    error_check( cudaDeviceSynchronize() )

    int tmp = 2*reflective_bc + 1*use_stencil_downsample;

    switch(tmp) {
        case 0:
            isotropic_convolve_555(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                    tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555);
            break;
        case 1:
            isotropic_convolve_555_ds(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                    tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, ne_rows_333_gpu.get(), ne_counter_333);
            break;
        case 2:
            isotropic_convolve_555_reflective(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                    tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555);
            break;
        case 3:
            isotropic_convolve_555_ds_reflective(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                    tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, ne_rows_333_gpu.get(), ne_counter_333);
            break;
        default:
            throw std::runtime_error("Error in isotropic_convolve_555: could not determine which set of kernels to launch");
    }
    error_check( cudaDeviceSynchronize() )

    /// transfer the results back to the host
    output_gpu.copyD2H();
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                                VectorData<stencilType>& stencil, VectorData<treeType>& tree_data, bool use_stencil_downsample, bool normalize_stencil) {

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 125);

    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());

    VectorData<stencilType> stencil_vec;
    if(use_stencil_downsample) {
        APRStencil::get_downsampled_stencils(stencil, stencil_vec, access.level_max() - access.level_min(), normalize_stencil);
    }

    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu;

    if(use_stencil_downsample) {
        stencil_gpu.initialize(stencil_vec.data(), stencil_vec.size());
    } else {
        stencil_gpu.initialize(stencil.data(), stencil.size());
    }

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    /// Fill the APR Tree by average downsampling
    downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    error_check( cudaDeviceSynchronize() )

    if(use_stencil_downsample) {
        isotropic_convolve_555_ds_alt(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
    } else {
        isotropic_convolve_555_alt(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
    }
    error_check( cudaDeviceSynchronize() )

    /// transfer the results back to the host
    output_gpu.copyD2H();
}


template void isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, bool, bool, bool);
template void isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, bool, bool, bool);
template void isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<double>&, bool, bool, bool);

template void isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, bool, bool);
template void isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&, bool, bool);
template void isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<double>&, bool, bool);
