//
// Created by cheesema on 05.04.18.
//

#include "APRDownsampleGPU.hpp"

#define DEBUGCUDA 1

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


template<typename inputType, typename outputType>
__global__ void _fill_tree_mean_max(const uint64_t* level_xz_vec,
                                    const uint64_t* xz_end_vec,
                                    const uint16_t* y_vec,
                                    const inputType* input_particles,
                                    const uint64_t* level_xz_vec_tree,
                                    const uint64_t* xz_end_vec_tree,
                                    const uint16_t* y_vec_tree,
                                    outputType* particle_data_output,
                                    const int z_num,
                                    const int x_num,
                                    const int y_num,
                                    const int z_num_parent,
                                    const int x_num_parent,
                                    const int y_num_parent,
                                    const int level,
                                    const int* offset_ind) {

    const int index = offset_ind[blockIdx.x];

    const int z_p = index/x_num_parent;
    const int x_p = index - z_p*x_num_parent;

    const int x_index = (2 * x_p + threadIdx.x/64);
    const int z_index = (2 * z_p + (threadIdx.x/32)%2);

    const int block = threadIdx.x/32;
    const int local_th = (threadIdx.x%32);

    __shared__ size_t global_index_begin_0_s[4];
    __shared__ size_t global_index_end_0_s[4];

    __shared__ float parent_cache[8][16];

    if(local_th < 16) {
        parent_cache[2 * block][local_th] = 0;
        parent_cache[2 * block + 1][local_th] = 0;
    }

    if( (x_index >= x_num) || (z_index >= z_num) ){
        return; //out of bounds
    }

    if((local_th==0) ) {
        size_t xz_start_s = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[block] = xz_end_vec[xz_start_s - 1];
        global_index_end_0_s[block] = xz_end_vec[xz_start_s];
    }

    __syncthreads();

    if(global_index_begin_0_s[0] == global_index_end_0_s[0]){
        return;
    }

    const size_t global_index_begin_0 = global_index_begin_0_s[block];
    const size_t global_index_end_0 = global_index_end_0_s[block];

    float current_val = 0;

    float scale_factor_xz = (((2*x_num_parent != x_num) && x_p==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && z_p==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    float scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }

    size_t xz_start = x_p + z_p*x_num_parent + level_xz_vec_tree[level-1];
    const size_t global_index_begin_p = xz_end_vec_tree[xz_start - 1];
    const size_t global_index_end_p = xz_end_vec_tree[xz_start];

    int current_y, current_y_p;

    if ((global_index_begin_0 + local_th) < global_index_end_0) {
        current_val = input_particles[global_index_begin_0 + local_th];
        current_y =  y_vec[global_index_begin_0 + local_th];
    } else {
        current_y = INT32_MAX;
    }

    if (block == 0) {
        if (( global_index_begin_p + local_th) < global_index_end_p) {
            current_y_p = y_vec_tree[global_index_begin_p + local_th];
        } else{
            current_y_p = INT32_MAX;
        }
    }

    const int block_start = y_vec[global_index_begin_0_s[0]] / 32;
    const int block_end = (y_vec[global_index_end_0_s[0] - 1] + 31) / 32;

    int sparse_block = 0;
    int sparse_block_p = 0;

    for (int y_block = block_start; y_block < block_end; ++y_block) {

        __syncthreads();
        //value less then current chunk then update.
        while(current_y < y_block * 32) {
            sparse_block++;
            if ((sparse_block * 32 + global_index_begin_0 + local_th) < global_index_end_0) {
                current_val = input_particles[sparse_block * 32 + global_index_begin_0 + local_th];
                current_y = y_vec[sparse_block * 32 + global_index_begin_0 + local_th];
            } else {
                current_y = INT32_MAX;
            }
        }

        __syncthreads();

        //update the down-sampling cache
        if ((current_y < (y_block + 1) * 32) && (current_y >= y_block * 32)) {
            parent_cache[2*block+current_y%2][(current_y/2) % 16] = (1.0f/8.0f)*current_val;
        }

        __syncthreads();
        //fetch the parent particle data
        if (block == 0) {
            while(current_y_p < ((y_block * 32)/2)) {
                sparse_block_p++;
                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {
                    current_y_p = y_vec_tree[sparse_block_p * 32 + global_index_begin_p + local_th];
                } else {
                    current_y_p = INT32_MAX;
                }
            }
        }
        __syncthreads();

        if(block == 0) {
            if ( (current_y_p < ((y_block+1) * 32)/2) ) {
                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {

                    if(current_y_p == (y_num_parent-1)) {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_yxz*( parent_cache[0][current_y_p % 16] +
                                                   parent_cache[1][current_y_p % 16] +
                                                   parent_cache[2][current_y_p % 16] +
                                                   parent_cache[3][current_y_p % 16] +
                                                   parent_cache[4][current_y_p % 16] +
                                                   parent_cache[5][current_y_p % 16] +
                                                   parent_cache[6][current_y_p % 16] +
                                                   parent_cache[7][current_y_p % 16]);

                    } else {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_xz*( parent_cache[0][current_y_p % 16] +
                                                  parent_cache[1][current_y_p % 16] +
                                                  parent_cache[2][current_y_p % 16] +
                                                  parent_cache[3][current_y_p % 16] +
                                                  parent_cache[4][current_y_p % 16] +
                                                  parent_cache[5][current_y_p % 16] +
                                                  parent_cache[6][current_y_p % 16] +
                                                  parent_cache[7][current_y_p % 16]);

                    }
                }
            }
        }
        __syncthreads();

        if(local_th < 16) {
            parent_cache[2 * block][local_th] = 0;
            parent_cache[2 * block + 1][local_th] = 0;
        }
    }
}


template<typename inputType, typename outputType>
__global__ void _fill_tree_mean_interior(const uint64_t* level_xz_vec,
                                         const uint64_t* xz_end_vec,
                                         const uint16_t* y_vec,
                                         const inputType* input_particles,
                                         const uint64_t* level_xz_vec_tree,
                                         const uint64_t* xz_end_vec_tree,
                                         const uint16_t* y_vec_tree,
                                         outputType* particle_data_output,
                                         const int z_num,
                                         const int x_num,
                                         const int y_num,
                                         const int z_num_parent,
                                         const int x_num_parent,
                                         const int y_num_parent,
                                         const int level,
                                         const int* offset_ind) {
    //
    //  This step is required for the interior down-sampling
    //

    const int index = offset_ind[blockIdx.x];

    const int z_p = index/x_num_parent;
    const int x_p = index - z_p*x_num_parent;

    //Local identifiers.
    int x_index = (2 * x_p + threadIdx.x/64);
    int z_index = (2 * z_p + ((threadIdx.x)/32)%2);

    const int block = threadIdx.x/32;
    const int local_th = (threadIdx.x%32);


    //Particles
    __shared__ std::size_t global_index_begin_0[4];
    __shared__ std::size_t global_index_end_0[4];

    //Parent Tree Particle Cells
    __shared__ std::size_t global_index_begin_p[4];
    __shared__ std::size_t global_index_end_p[4];

    //Tree Particle Cells
    __shared__ std::size_t global_index_begin_t[4];
    __shared__ std::size_t global_index_end_t[4];

    //shared memory caches
    __shared__ float parent_cache[8][16]; //16 needed padded with 17 entries to optimize for bank conflicts.

    if(local_th < 16) {
        parent_cache[2 * block][local_th] = 0;
        parent_cache[2 * block + 1][local_th] = 0;
    }

    if((x_index >= x_num) || (z_index >= z_num) ){
        return;
    }

    if(local_th == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t[block] = xz_end_vec_tree[xz_start];
    }

    if(local_th == 1) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0[block] = xz_end_vec[xz_start - 1];
        global_index_end_0[block] = xz_end_vec[xz_start];
    }

    if(local_th == 2) {
        size_t xz_start = x_p + z_p * x_num_parent + level_xz_vec_tree[level - 1];
        global_index_begin_p[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_p[block] = xz_end_vec_tree[xz_start];
    }

    __syncthreads();

    if((global_index_begin_0[block] == global_index_end_0[block]) && (global_index_begin_t[block] == global_index_end_t[block])){
        return;
    }

    float scale_factor_xz = (((2*x_num_parent != x_num) && x_p==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && z_p==(z_num_parent-1) ))*2;
    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    float scale_factor_yxz = scale_factor_xz;
    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }

    int y_0, y_p, y_t;
    float f_0, f_t;

    __syncthreads();
    //initialize (i=0)
    if ((global_index_begin_0[block] + local_th) < global_index_end_0[block]) {
        y_0 = y_vec[global_index_begin_0[block] + local_th];
        f_0 = input_particles[global_index_begin_0[block] + local_th];
    } else {
        y_0 = INT32_MAX;
    }

    __syncthreads();
    //tree interior
    if ((global_index_begin_t[block] + local_th) < global_index_end_t[block]) {
        y_t = y_vec_tree[global_index_begin_t[block] + local_th];
        f_t = particle_data_output[global_index_begin_t[block] + local_th];
    } else {
        y_t = INT32_MAX;
    }

    __syncthreads();
    if (block == 0) {
        if (( global_index_begin_p[block] + local_th) < global_index_end_p[block]) {
            y_p = y_vec_tree[global_index_begin_p[block] + local_th];
        } else {
            y_p = INT32_MAX;
        }
    }

    __shared__ int block_start[4];
    __shared__ int block_end[4];

    // initialize indices in case some blocks stopped early
    __syncthreads();
    if( (block == 0) && (local_th < 4) ) {
        block_start[local_th] = INT32_MAX;
        block_end[local_th] = 0;
    }

    __syncthreads();

    if(local_th == 0) {
        block_start[block] = min(y_vec[global_index_begin_0[block]],  y_vec_tree[global_index_begin_t[block]]) / 32;
        block_end[block] = (max(y_vec[max(global_index_end_0[block],(size_t)1)-1], y_vec_tree[ max(global_index_end_t[block],(size_t)1)-1]) + 31)/32;
    }

    __syncthreads();

    if( (block == 0) && (local_th == 0) ) {
        block_start[0] = min( min(block_start[0], block_start[1]), min(block_start[2], block_start[3]) );
        block_end[0] = max( max(block_end[0], block_end[1]), max(block_end[2], block_end[3]) );
    }

    __syncthreads();
    int sparse_block = 0;
    int sparse_block_p = 0;
    int sparse_block_t = 0;

    for (int y_block = block_start[0]; y_block < block_end[0]; ++y_block) {
//    for (int y_block = 0; y_block < (y_num+31)/32; y_block++) {
        __syncthreads();

        // update apr particle
        while(y_0 < (y_block * 32)) {
            sparse_block++;
            if ((sparse_block * 32 + global_index_begin_0[block] + local_th) < global_index_end_0[block]) {

                f_0 = input_particles[sparse_block * 32 + global_index_begin_0[block] + local_th];
                y_0 = y_vec[sparse_block * 32 + global_index_begin_0[block] + local_th];
            } else{
                y_0 = INT32_MAX;
            }
        }

        __syncthreads();

        // update tree particle
        while(y_t < (y_block * 32)) {
            sparse_block_t++;
            if ((sparse_block_t * 32 + global_index_begin_t[block] + local_th) < global_index_end_t[block]) {

                f_t = particle_data_output[sparse_block_t * 32 + global_index_begin_t[block] + local_th];
                y_t = y_vec_tree[sparse_block_t * 32 + global_index_begin_t[block] + local_th];
            } else{
                y_t = INT32_MAX;
            }
        }

        __syncthreads();

        ///update the down-sampling cache
        //insert apr particles
        if (y_0 < (y_block + 1) * 32) {
            parent_cache[2*block + y_0 % 2][(y_0 / 2) % 16] = (1.0f / 8.0f) * f_0;
        }

        __syncthreads();

        //insert tree particles
        if (y_t < (y_block + 1) * 32) {
            parent_cache[2*block + y_t % 2][(y_t / 2) % 16] = (1.0f / 8.0f) * f_t;
        }
        __syncthreads();

        // update parent particle
        if (block == 0) {
            while(y_p < ((y_block * 32) / 2)) {
                sparse_block_p++;

                if ((sparse_block_p * 32 + global_index_begin_p[block] + local_th) < global_index_end_p[block]) {
                    y_p = y_vec_tree[sparse_block_p * 32 + global_index_begin_p[block] + local_th];
                } else{
                    y_p = INT32_MAX;
                }
            }
        }

        __syncthreads();

        // perform the reduction and write result to output array
        if(block == 0) {

            if (y_p < ((y_block + 1) * 32) / 2) {   //current_y_p >= ((y_block) * 32)/2 is guaranteed from update step
                if ((sparse_block_p * 32 + global_index_begin_p[block] + local_th) < global_index_end_p[block]) {

                    if (y_p == (y_num_parent - 1)) {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p[block] + local_th] =
                                scale_factor_yxz * (parent_cache[0][y_p % 16] +
                                                    parent_cache[1][y_p % 16] +
                                                    parent_cache[2][y_p % 16] +
                                                    parent_cache[3][y_p % 16] +
                                                    parent_cache[4][y_p % 16] +
                                                    parent_cache[5][y_p % 16] +
                                                    parent_cache[6][y_p % 16] +
                                                    parent_cache[7][y_p % 16]);

                    } else {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p[block] + local_th] =
                                scale_factor_xz * (parent_cache[0][y_p % 16] +
                                                   parent_cache[1][y_p % 16] +
                                                   parent_cache[2][y_p % 16] +
                                                   parent_cache[3][y_p % 16] +
                                                   parent_cache[4][y_p % 16] +
                                                   parent_cache[5][y_p % 16] +
                                                   parent_cache[6][y_p % 16] +
                                                   parent_cache[7][y_p % 16]);

                    }
                }
            }
        }

        __syncthreads();

        // reset the cache
        if(local_th < 16) {
            parent_cache[2 * block][local_th] = 0;
            parent_cache[2 * block + 1][local_th] = 0;
        }
    }
}


template<typename inputType, typename outputType>
__global__ void _fill_tree_mean_max_alt(const uint64_t* level_xz_vec,
                                        const uint64_t* xz_end_vec,
                                        const uint16_t* y_vec,
                                        const inputType* input_particles,
                                        const uint64_t* level_xz_vec_tree,
                                        const uint64_t* xz_end_vec_tree,
                                        const uint16_t* y_vec_tree,
                                        outputType* particle_data_output,
                                        const int z_num,
                                        const int x_num,
                                        const int y_num,
                                        const int z_num_parent,
                                        const int x_num_parent,
                                        const int y_num_parent,
                                        const int level) {

    const int z_index = blockIdx.z * blockDim.z + threadIdx.z;
    const int x_index = blockIdx.x * blockDim.y + threadIdx.y;

    const int block = threadIdx.z * 2 + threadIdx.y;
    const int local_th = threadIdx.x;

    __shared__ float parent_cache[8][16];

    if(local_th < 16) {
        parent_cache[2 * block][local_th] = 0;
        parent_cache[2 * block + 1][local_th] = 0;
    }

    if( (x_index >= x_num) || (z_index >= z_num) ){
        return; //out of bounds
    }

    __shared__ size_t global_index_begin_0_s[4];
    __shared__ size_t global_index_end_0_s[4];

    if((local_th==0) ) {
        size_t xz_start_s = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0_s[block] = xz_end_vec[xz_start_s - 1];
        global_index_end_0_s[block] = xz_end_vec[xz_start_s];
    }

    __syncthreads();

    if(global_index_begin_0_s[0] == global_index_end_0_s[0]){
        return;
    }

    const size_t global_index_begin_0 = global_index_begin_0_s[block];
    const size_t global_index_end_0 = global_index_end_0_s[block];

    float scale_factor_xz = (((2*x_num_parent != x_num) && (x_index / 2)==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && (z_index / 2)==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    float scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }

    size_t xz_start = (x_index / 2) + (z_index / 2)*x_num_parent + level_xz_vec_tree[level-1];
    const size_t global_index_begin_p = xz_end_vec_tree[xz_start - 1];
    const size_t global_index_end_p = xz_end_vec_tree[xz_start];

    int current_y;
    int current_y_p;
    float current_val = 0;

    //initialize (i=0)
    if ((global_index_begin_0 + local_th) < global_index_end_0) {
        current_val = input_particles[global_index_begin_0 + local_th];
        current_y =  y_vec[global_index_begin_0 + local_th];
    } else {
        current_y = INT32_MAX;
    }

    if (block == 0) {
        if (( global_index_begin_p + local_th) < global_index_end_p) {
            current_y_p = y_vec_tree[global_index_begin_p + local_th];
        } else{
            current_y_p = INT32_MAX;
        }
    }

    const int block_start = y_vec[global_index_begin_0_s[0]] / 32;
    const int block_end = (y_vec[global_index_end_0_s[0]-1] + 31) / 32;

    int sparse_block = 0;
    int sparse_block_p = 0;

    for (int y_block = block_start; y_block < block_end; ++y_block) {

        __syncthreads();
        //value less then current chunk then update.
        while(current_y < y_block * 32) {
            sparse_block++;
            if ((sparse_block * 32 + global_index_begin_0 + local_th) < global_index_end_0) {
                current_val = input_particles[sparse_block * 32 + global_index_begin_0 + local_th];
                current_y = y_vec[sparse_block * 32 + global_index_begin_0 + local_th];
            } else{
                current_y = INT32_MAX;
            }
        }

        __syncthreads();

        //update the down-sampling caches
        if ((current_y < (y_block + 1) * 32) && (current_y >= (y_block) * 32)) {
            parent_cache[2*block+current_y%2][(current_y/2) % 16] = (1.0f/8.0f)*current_val;
        }

        __syncthreads();
        //fetch the parent particle data
        if (block == 0) {
            while(current_y_p < ((y_block * 32)/2)) {
                sparse_block_p++;
                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {
                    current_y_p = y_vec_tree[sparse_block_p * 32 + global_index_begin_p + local_th];
                } else {
                    current_y_p = INT32_MAX;
                }
            }
        }
        __syncthreads();

        if(block == 0) {
            if (current_y_p < ((y_block+1) * 32)/2) {
                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {

                    if(current_y_p == (y_num_parent-1)) {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_yxz*( parent_cache[0][current_y_p % 16] +
                                                   parent_cache[1][current_y_p % 16] +
                                                   parent_cache[2][current_y_p % 16] +
                                                   parent_cache[3][current_y_p % 16] +
                                                   parent_cache[4][current_y_p % 16] +
                                                   parent_cache[5][current_y_p % 16] +
                                                   parent_cache[6][current_y_p % 16] +
                                                   parent_cache[7][current_y_p % 16]);

                    } else {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_xz*( parent_cache[0][current_y_p % 16] +
                                                  parent_cache[1][current_y_p % 16] +
                                                  parent_cache[2][current_y_p % 16] +
                                                  parent_cache[3][current_y_p % 16] +
                                                  parent_cache[4][current_y_p % 16] +
                                                  parent_cache[5][current_y_p % 16] +
                                                  parent_cache[6][current_y_p % 16] +
                                                  parent_cache[7][current_y_p % 16]);

                    }
                }
            }
        }
        __syncthreads();
        if(local_th < 16) {
            parent_cache[2 * block][local_th] = 0;
            parent_cache[2 * block + 1][local_th] = 0;
        }
    }
}


template<typename inputType, typename outputType>
__global__ void _fill_tree_mean_interior_alt(const uint64_t* level_xz_vec,
                                             const uint64_t* xz_end_vec,
                                             const uint16_t* y_vec,
                                             const inputType* input_particles,
                                             const uint64_t* level_xz_vec_tree,
                                             const uint64_t* xz_end_vec_tree,
                                             const uint16_t* y_vec_tree,
                                             outputType* particle_data_output,
                                             const int z_num,
                                             const int x_num,
                                             const int y_num,
                                             const int z_num_parent,
                                             const int x_num_parent,
                                             const int y_num_parent,
                                             const int level) {
    //
    //  This step is required for the interior down-sampling
    //

    const int z_index = blockIdx.z * blockDim.z + threadIdx.z;
    const int x_index = blockIdx.x * blockDim.y + threadIdx.y;

    const int block = threadIdx.z * 2 + threadIdx.y;
    const int local_th = threadIdx.x;

    //shared memory cache
    __shared__ float parent_cache[8][16]; //16 needed padded with 17 entries to optimize for bank conflicts.

    if(local_th < 16) {
        parent_cache[2 * block][local_th] = 0;
        parent_cache[2 * block + 1][local_th] = 0;
    }

    if((x_index >= x_num) || (z_index >= z_num) ){
        return;
    }

    //Particles
    __shared__ std::size_t global_index_begin_0[4];
    __shared__ std::size_t global_index_end_0[4];

    //Parent Tree Particle Cells
    __shared__ std::size_t global_index_begin_p[4];
    __shared__ std::size_t global_index_end_p[4];

    //Interior Tree Particle Cells
    __shared__ std::size_t global_index_begin_t[4];
    __shared__ std::size_t global_index_end_t[4];

    if(local_th == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t[block] = xz_end_vec_tree[xz_start];
    }

    if(local_th == 1) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0[block] = xz_end_vec[xz_start - 1];
        global_index_end_0[block] = xz_end_vec[xz_start];
    }

    if(local_th == 2) {
        size_t xz_start = (x_index / 2) + (z_index / 2) * x_num_parent + level_xz_vec_tree[level - 1];
        global_index_begin_p[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_p[block] = xz_end_vec_tree[xz_start];
    }

    __syncthreads();

    if((global_index_begin_0[block] == global_index_end_0[block]) && (global_index_begin_t[block] == global_index_end_t[block])){
        return;
    }

    float scale_factor_xz = (((2*x_num_parent != x_num) && (x_index / 2)==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && (z_index / 2)==(z_num_parent-1) ))*2;
    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    float scale_factor_yxz = scale_factor_xz;
    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }

    int y_0, y_p, y_t;
    float f_0, f_t;

    __syncthreads();
    //each thread grabs a particle
    //from the apr
    if ((global_index_begin_0[block] + local_th) < global_index_end_0[block]) {
        y_0 = y_vec[global_index_begin_0[block] + local_th];
        f_0 = input_particles[global_index_begin_0[block] + local_th];
    } else {
        y_0 = INT32_MAX;
    }

    __syncthreads();
    //from the tree
    if ((global_index_begin_t[block] + local_th) < global_index_end_t[block]) {
        y_t = y_vec_tree[global_index_begin_t[block] + local_th];
        f_t = particle_data_output[global_index_begin_t[block] + local_th];
    } else {
        y_t = INT32_MAX;
    }

    __syncthreads();
    //parent particle (tree)
    if (block == 0) {
        if (( global_index_begin_p[block] + local_th) < global_index_end_p[block]) {
            y_p = y_vec_tree[global_index_begin_p[block] + local_th];
        } else {
            y_p = INT32_MAX;
        }
    }

    __shared__ int block_start[4];
    __shared__ int block_end[4];

    // initialize indices in case some blocks stopped early
    __syncthreads();
    if( (block == 0) && (local_th < 4) ) {
        block_start[local_th] = INT32_MAX;
        block_end[local_th] = 0;
    }

    __syncthreads();

    if(local_th == 0) {
        block_start[block] = min(y_vec[global_index_begin_0[block]], y_vec_tree[global_index_begin_t[block]]) / 32;
        block_end[block] = ((int)max(y_vec[(size_t)max(global_index_end_0[block],(size_t)1)-1], y_vec_tree[(size_t)max(global_index_end_t[block],(size_t)1)-1]) + 31) / 32;
    }

    __syncthreads();

    if( (block == 0) && (local_th == 0) ) {
        block_start[0] = min(min(block_start[0], block_start[1]), min(block_start[2], block_start[3]) );
        block_end[0] = max(max(block_end[0], block_end[1]), max(block_end[2], block_end[3]));
    }

    __syncthreads();
    int sparse_block = 0;
    int sparse_block_p = 0;
    int sparse_block_t = 0;

    for (int y_block = block_start[0]; y_block < block_end[0]; ++y_block) {

        __syncthreads();
        //value less then current chunk then update.
        while(y_0 < (y_block * 32)) {
            sparse_block++;
            if ((sparse_block * 32 + global_index_begin_0[block] + local_th) < global_index_end_0[block]) {

                f_0 = input_particles[sparse_block * 32 + global_index_begin_0[block] + local_th];
                y_0 = y_vec[sparse_block * 32 + global_index_begin_0[block] + local_th];
            } else{
                y_0 = INT32_MAX;
            }
        }

        __syncthreads();
        //interior tree update
        while(y_t < (y_block * 32)) {
            sparse_block_t++;
            if ((sparse_block_t * 32 + global_index_begin_t[block] + local_th) < global_index_end_t[block]) {

                f_t = particle_data_output[sparse_block_t * 32 + global_index_begin_t[block] + local_th];
                y_t = y_vec_tree[sparse_block_t * 32 + global_index_begin_t[block] + local_th];
            } else{
                y_t = INT32_MAX;
            }
        }

        __syncthreads();
        //update the down-sampling caches
        if( y_0 < (y_block + 1) * 32 ) {
            parent_cache[2*block + y_0 % 2][(y_0 / 2) % 16] = (1.0f / 8.0f) * f_0;
        }
        __syncthreads();

        //now the interior tree nodes
        if ( y_t < (y_block + 1) * 32 ) {
            parent_cache[2*block + y_t % 2][(y_t / 2) % 16] = (1.0f / 8.0f) * f_t;
        }
        __syncthreads();

        if (block == 0) {
            while(y_p < ((y_block * 32) / 2)) {
                sparse_block_p++;
                if ((sparse_block_p * 32 + global_index_begin_p[block] + local_th) < global_index_end_p[block]) {
                    y_p = y_vec_tree[sparse_block_p * 32 + global_index_begin_p[block] + local_th];
                } else {
                    y_p = INT32_MAX;
                }
            }
        }

        __syncthreads();

        if(block == 0) {
            if ( y_p < ((y_block + 1) * 32) / 2 ) {
                if ((sparse_block_p * 32 + global_index_begin_p[block] + local_th) < global_index_end_p[block]) {

                    if (y_p == (y_num_parent - 1)) {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p[block] + local_th] =
                                scale_factor_yxz * (parent_cache[0][y_p % 16] +
                                                    parent_cache[1][y_p % 16] +
                                                    parent_cache[2][y_p % 16] +
                                                    parent_cache[3][y_p % 16] +
                                                    parent_cache[4][y_p % 16] +
                                                    parent_cache[5][y_p % 16] +
                                                    parent_cache[6][y_p % 16] +
                                                    parent_cache[7][y_p % 16]);

                    } else {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p[block] + local_th] =
                                scale_factor_xz * (parent_cache[0][y_p % 16] +
                                                   parent_cache[1][y_p % 16] +
                                                   parent_cache[2][y_p % 16] +
                                                   parent_cache[3][y_p % 16] +
                                                   parent_cache[4][y_p % 16] +
                                                   parent_cache[5][y_p % 16] +
                                                   parent_cache[6][y_p % 16] +
                                                   parent_cache[7][y_p % 16]);

                    }
                }
            }
        }

        __syncthreads();

        if(local_th < 16) {
            parent_cache[2 * block][local_th] = 0;
            parent_cache[2 * block + 1][local_th] = 0;
        }
    }
}



void compute_ne_rows(GPUAccessHelper& tree_access,VectorData<int>& ne_counter,VectorData<int>& ne_rows) {
    ne_counter.resize(tree_access.level_max() + 3);

    int z = 0;
    int x = 0;

    uint64_t counter = 0;

    for (int level = (tree_access.level_min() + 1); level <= (tree_access.level_max() + 1); ++level) {

        auto level_start = tree_access.linearAccess->level_xz_vec[level - 1];

        ne_counter[level] = counter;

        for (z = 0; z < tree_access.z_num(level - 1); z++) {
            for (x = 0; x < tree_access.x_num(level - 1); ++x) {

                auto offset = x + z * tree_access.x_num(level - 1);
                auto xz_start = level_start + offset;

                auto begin_index = tree_access.linearAccess->xz_end_vec[xz_start - 1];
                auto end_index = tree_access.linearAccess->xz_end_vec[xz_start];

                if (begin_index < end_index) {
                    counter++;
                }
            }
        }
    }

    ne_rows.resize(counter);
    ne_counter.back() = ne_rows.size();
    counter = 0;

    for (int level = (tree_access.level_min() + 1); level <= (tree_access.level_max() + 1); ++level) {

        auto level_start = tree_access.linearAccess->level_xz_vec[level - 1];

        for (z = 0; z < tree_access.z_num(level - 1); z++) {
            for (x = 0; x < tree_access.x_num(level - 1); ++x) {

                auto offset = x + z * tree_access.x_num(level - 1);
                auto xz_start = level_start + offset;

//intialize
                auto begin_index = tree_access.linearAccess->xz_end_vec[xz_start - 1];
                auto end_index = tree_access.linearAccess->xz_end_vec[xz_start];

                if (begin_index < end_index) {
                    ne_rows[counter] = (x + z * tree_access.x_num(level - 1));
                    counter++;
                }
            }
        }
    }
}


template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu,int* ne_rows,VectorData<int>& ne_offset) {

    /// assumes input_gpu, tree_data_gpu and ne_rows are already on the device

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        if(level == access.level_max()){

            size_t ne_sz = ne_offset[level+1] - ne_offset[level];
            size_t offset = ne_offset[level];

            dim3 threads_l(128, 1, 1);
            dim3 blocks_l(ne_sz, 1, 1);

            _fill_tree_mean_max << < blocks_l, threads_l >> >
                                               (access.get_level_xz_vec_ptr(),
                                                   access.get_xz_end_vec_ptr(),
                                                   access.get_y_vec_ptr(),
                                                   input_gpu,
                                                   tree_access.get_level_xz_vec_ptr(),
                                                   tree_access.get_xz_end_vec_ptr(),
                                                   tree_access.get_y_vec_ptr(),
                                                   tree_data_gpu,
                                                   access.z_num(level),
                                                   access.x_num(level),
                                                   access.y_num(level),
                                                   tree_access.z_num(level-1),
                                                   tree_access.x_num(level-1),
                                                   tree_access.y_num(level-1),
                                                   level,
                                                   ne_rows + offset);


        } else {

            dim3 threads_l(128, 1, 1);

            size_t ne_sz = ne_offset[level+1] - ne_offset[level];
            size_t offset = ne_offset[level];

            dim3 blocks_l(ne_sz, 1, 1);

            _fill_tree_mean_interior << < blocks_l, threads_l >> >
                                                    (access.get_level_xz_vec_ptr(),
                                                       access.get_xz_end_vec_ptr(),
                                                       access.get_y_vec_ptr(),
                                                       input_gpu,
                                                       tree_access.get_level_xz_vec_ptr(),
                                                       tree_access.get_xz_end_vec_ptr(),
                                                       tree_access.get_y_vec_ptr(),
                                                       tree_data_gpu,
                                                       access.z_num(level),
                                                       access.x_num(level),
                                                       access.y_num(level),
                                                       tree_access.z_num(level-1),
                                                       tree_access.x_num(level-1),
                                                       tree_access.y_num(level-1),
                                                       level,
                                                       ne_rows + offset);

        }

        error_check( cudaDeviceSynchronize() )
    }
}


template<typename inputType, typename treeType>
void downsample_avg_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu) {

    /// assumes that access structures, input_gpu and tree_data_gpu are already on the device

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        int x_blocks = (access.x_num(level) + 1) / 2;
        int z_blocks = (access.z_num(level) + 1) / 2;

        dim3 blocks_l(x_blocks, 1, z_blocks);
        dim3 threads_l(32, 2, 2);

        if(level == access.level_max()){

            _fill_tree_mean_max_alt << < blocks_l, threads_l >> >
                                               (access.get_level_xz_vec_ptr(),
                                               access.get_xz_end_vec_ptr(),
                                               access.get_y_vec_ptr(),
                                               input_gpu,
                                               tree_access.get_level_xz_vec_ptr(),
                                               tree_access.get_xz_end_vec_ptr(),
                                               tree_access.get_y_vec_ptr(),
                                               tree_data_gpu,
                                               access.z_num(level),
                                               access.x_num(level),
                                               access.y_num(level),
                                               tree_access.z_num(level-1),
                                               tree_access.x_num(level-1),
                                               tree_access.y_num(level-1),
                                               level);
        } else {

            _fill_tree_mean_interior_alt << < blocks_l, threads_l >> >
                                                    (access.get_level_xz_vec_ptr(),
                                                    access.get_xz_end_vec_ptr(),
                                                    access.get_y_vec_ptr(),
                                                    input_gpu,
                                                    tree_access.get_level_xz_vec_ptr(),
                                                    tree_access.get_xz_end_vec_ptr(),
                                                    tree_access.get_y_vec_ptr(),
                                                    tree_data_gpu,
                                                    access.z_num(level),
                                                    access.x_num(level),
                                                    access.y_num(level),
                                                    tree_access.z_num(level-1),
                                                    tree_access.x_num(level-1),
                                                    tree_access.y_num(level-1),
                                                    level);
        }

        error_check( cudaDeviceSynchronize() )
    }
}


template<typename inputType, typename treeType>
void downsample_avg_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<treeType>& tree_data) {

    if(tree_data.size() != tree_access.total_number_particles()) {
        tree_data.resize(tree_access.total_number_particles());
    }
    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());

    input_gpu.copyH2D();

    downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());

    tree_data_gpu.copyD2H();
}


template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu) {

    VectorData<int> ne_counter;
    VectorData<int> ne_rows;
    compute_ne_rows(tree_access, ne_counter, ne_rows);

    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu(ne_rows.data(), ne_rows.size());
    ne_rows_gpu.copyH2D();

    downsample_avg(access, tree_access, input_gpu, tree_data_gpu, ne_rows_gpu.get(), ne_counter);
}


template<typename inputType, typename treeType>
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<treeType>& tree_data) {

    if(tree_data.size() != tree_access.total_number_particles()) {
        tree_data.resize(tree_access.total_number_particles());
    }
    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());

    VectorData<int> ne_counter;
    VectorData<int> ne_rows;
    compute_ne_rows(tree_access, ne_counter, ne_rows);

    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu(ne_rows.data(), ne_rows.size());
    ne_rows_gpu.copyH2D();

    input_gpu.copyH2D();

    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);

    tree_data_gpu.copyD2H();
}

/// initialize templates
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<uint16_t>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<double>&);

template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, uint16_t*);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, float*);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, double*);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, float*, float*);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, float*, double*);

template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, uint16_t*,int*,VectorData<int>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, float*,int*,VectorData<int>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, double*,int*,VectorData<int>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, float*, float*,int*,VectorData<int>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, float*, double*,int*,VectorData<int>&);

template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<uint16_t>&);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<double>&);

template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, uint16_t*);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, float*);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, double*);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, float*, float*);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, float*, double*);
