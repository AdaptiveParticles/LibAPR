//
// Created by cheesema on 05.04.18.
//

#include "APRDownsampleGPU.hpp"


template<typename inputType, typename outputType>
__global__ void _fill_tree_mean_max(const uint64_t* __restrict__ level_xz_vec,
                                    const uint64_t* __restrict__ xz_end_vec,
                                    const uint16_t* __restrict__ y_vec,
                                    const inputType* __restrict__ input_particles,
                                    const uint64_t* __restrict__ level_xz_vec_tree,
                                    const uint64_t* __restrict__ xz_end_vec_tree,
                                    const uint16_t* __restrict__ y_vec_tree,
                                    outputType* __restrict__ particle_data_output,
                                    const int z_num,
                                    const int x_num,
                                    const int y_num,
                                    const int z_num_parent,
                                    const int x_num_parent,
                                    const int y_num_parent,
                                    const int level,
                                    const int* __restrict__ offset_ind) {

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

    __syncwarp();
    if ((global_index_begin_0 + local_th) < global_index_end_0) {
        current_val = input_particles[global_index_begin_0 + local_th];
        current_y =  y_vec[global_index_begin_0 + local_th];
    } else {
        current_y = INT32_MAX;
    }
    __syncwarp();

    if (block == 0) {
        if (( global_index_begin_p + local_th) < global_index_end_p) {
            current_y_p = y_vec_tree[global_index_begin_p + local_th];
        } else{
            current_y_p = INT32_MAX;
        }
    }
    __syncwarp();


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

        __syncwarp();

        //update the down-sampling cache
        if ((current_y < (y_block + 1) * 32) && (current_y >= y_block * 32)) {
            parent_cache[2*block+current_y%2][(current_y/2) % 16] = (1.0f/8.0f)*current_val;
        }

        __syncwarp();
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
__global__ void _fill_tree_mean_interior(const uint64_t* __restrict__ level_xz_vec,
                                         const uint64_t* __restrict__ xz_end_vec,
                                         const uint16_t* __restrict__ y_vec,
                                         const inputType* __restrict__ input_particles,
                                         const uint64_t* __restrict__ level_xz_vec_tree,
                                         const uint64_t* __restrict__ xz_end_vec_tree,
                                         const uint16_t* __restrict__ y_vec_tree,
                                         outputType* __restrict__ particle_data_output,
                                         const int z_num,
                                         const int x_num,
                                         const int y_num,
                                         const int z_num_parent,
                                         const int x_num_parent,
                                         const int y_num_parent,
                                         const int level,
                                         const int* __restrict__ offset_ind) {
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
    __syncwarp();


    if((x_index >= x_num) || (z_index >= z_num) ){
        return;
    }

    if(local_th == 0) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t[block] = xz_end_vec_tree[xz_start - 1];
        global_index_end_t[block] = xz_end_vec_tree[xz_start];
    }
    __syncwarp();


    if(local_th == 1) {
        size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0[block] = xz_end_vec[xz_start - 1];
        global_index_end_0[block] = xz_end_vec[xz_start];
    }
    __syncwarp();


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

    __syncwarp();
    if ((global_index_begin_0[block] + local_th) < global_index_end_0[block]) {
        y_0 = y_vec[global_index_begin_0[block] + local_th];
        f_0 = input_particles[global_index_begin_0[block] + local_th];
    } else {
        y_0 = INT32_MAX;
    }

    __syncwarp();
    if ((global_index_begin_t[block] + local_th) < global_index_end_t[block]) {
        y_t = y_vec_tree[global_index_begin_t[block] + local_th];
        f_t = particle_data_output[global_index_begin_t[block] + local_th];
    } else {
        y_t = INT32_MAX;
    }

    __syncwarp();
    if (block == 0) {
        if (( global_index_begin_p[block] + local_th) < global_index_end_p[block]) {
            y_p = y_vec_tree[global_index_begin_p[block] + local_th];
        } else {
            y_p = INT32_MAX;
        }
    }
    __syncwarp();


    const int block_start = y_vec_tree[global_index_begin_p[0]] / 16;
    const int block_end = ((2 * y_vec_tree[max(global_index_end_p[0], (size_t)1) - 1] + 32) / 32); // "ceil( (2 * y_tree + 1) / 32 )"

    int sparse_block = 0;
    int sparse_block_p = 0;
    int sparse_block_t = 0;

    for (int y_block = block_start; y_block < block_end; ++y_block) {
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
        __syncwarp();

        ///update the down-sampling cache
        //insert apr particles
        if (y_0 < (y_block + 1) * 32) {
            parent_cache[2*block + y_0 % 2][(y_0 / 2) % 16] = (1.0f / 8.0f) * f_0;
        }
        __syncwarp();

        //insert tree particles
        if (y_t < (y_block + 1) * 32) {
            parent_cache[2*block + y_t % 2][(y_t / 2) % 16] = (1.0f / 8.0f) * f_t;
        }
        __syncwarp();

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
__global__ void _fill_tree_mean_max_alt(const uint64_t* __restrict__ level_xz_vec,
                                        const uint64_t* __restrict__ xz_end_vec,
                                        const uint16_t* __restrict__ y_vec,
                                        const inputType* __restrict__ input_particles,
                                        const uint64_t* __restrict__ level_xz_vec_tree,
                                        const uint64_t* __restrict__ xz_end_vec_tree,
                                        const uint16_t* __restrict__ y_vec_tree,
                                        outputType* __restrict__ particle_data_output,
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
__global__ void _fill_tree_mean_interior_alt(const uint64_t* __restrict__ level_xz_vec,
                                             const uint64_t* __restrict__ xz_end_vec,
                                             const uint16_t* __restrict__ y_vec,
                                             const inputType* __restrict__ input_particles,
                                             const uint64_t* __restrict__ level_xz_vec_tree,
                                             const uint64_t* __restrict__ xz_end_vec_tree,
                                             const uint16_t* __restrict__ y_vec_tree,
                                             outputType* __restrict__ particle_data_output,
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

    const int block_start = y_vec_tree[global_index_begin_p[0]] / 16;
    const int block_end = ((2 * y_vec_tree[max(global_index_end_p[0], (size_t)1) - 1] + 32) / 32); // "ceil( (2 * y_tree + 1) / 32 )"

    int sparse_block = 0;
    int sparse_block_p = 0;
    int sparse_block_t = 0;

    for (int y_block = block_start; y_block < block_end; ++y_block) {

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


template<int blockSize_z, int blockSize_x>
__global__ void _count_ne_rows_tree_cuda(const uint64_t* __restrict__ level_xz_vec_tree,
                                         const uint64_t* __restrict__ xz_end_vec_tree,
                                         const int z_num,
                                         const int x_num,
                                         const int level,
                                         int* __restrict__ res) {

    __shared__ int local_counts[blockSize_x][blockSize_z];
    local_counts[threadIdx.y][threadIdx.x] = 0;

    const int z_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(z_index >= z_num) { return; } // out of bounds

    size_t level_start = level_xz_vec_tree[level];
    int x_index = threadIdx.y;

    int counter = 0;

    // loop over x-dimension in chunks
    while( x_index < x_num ) {
        size_t xz_start = z_index * x_num + x_index + level_start;

        // if row is non-empty
        if( xz_end_vec_tree[xz_start - 1] < xz_end_vec_tree[xz_start]) {
            counter++;
        }
        x_index += blockDim.y;
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
__global__ void _fill_ne_rows_tree_cuda(const uint64_t* __restrict__ level_xz_vec_tree,
                                        const uint64_t* __restrict__ xz_end_vec_tree,
                                        const int z_num,
                                        const int x_num,
                                        const int level,
                                        unsigned int ne_count,
                                        unsigned int offset,
                                        int* __restrict__ ne_rows) {

    const int z_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (z_index >= z_num) { return; } // out of bounds

    size_t level_start = level_xz_vec_tree[level];
    int x_index = threadIdx.y;

    // loop over x-dimension in chunks
    while (x_index < x_num) {

        size_t xz_start = z_index * x_num + x_index + level_start;

        // if row is non-empty
        if( xz_end_vec_tree[xz_start - 1] < xz_end_vec_tree[xz_start]) {
            unsigned int index = atomicInc(&count, ne_count-1);
            ne_rows[offset + index] = z_index * x_num + x_index;
        }

        x_index += blockDim.y;
    }
}


template<int blockSize_z, int blockSize_x>
void compute_ne_rows_tree_cuda(GPUAccessHelper& tree_access, VectorData<int>& ne_count, ScopedCudaMemHandler<int*, JUST_ALLOC>& ne_rows_gpu) {

    ne_count.resize(tree_access.level_max() + 3);
    ne_count[0] = 0;

    int z_blocks_max = (tree_access.z_num(tree_access.level_max()) + blockSize_z - 1) / blockSize_z;
    int num_levels = tree_access.level_max() - tree_access.level_min() + 1;

    int block_sums_host[z_blocks_max * num_levels];
    int *block_sums_device;

    error_check(cudaMalloc(&block_sums_device, z_blocks_max*num_levels*sizeof(int)) )
    error_check( cudaMemset(block_sums_device, 0, z_blocks_max*num_levels*sizeof(int)) )
//    error_check( cudaDeviceSynchronize() )

    int offset = 0;
    for(int level = tree_access.level_min(); level <= tree_access.level_max(); ++level) {

        int z_blocks = (tree_access.z_num(level) + blockSize_z - 1) / blockSize_z;

        dim3 grid_dim(z_blocks, 1, 1);
        dim3 block_dim(blockSize_z, blockSize_x, 1);

        _count_ne_rows_tree_cuda<blockSize_z, blockSize_x>
                                << < grid_dim, block_dim >> >
                                    (tree_access.get_level_xz_vec_ptr(),
                                     tree_access.get_xz_end_vec_ptr(),
                                     tree_access.z_num(level),
                                     tree_access.x_num(level),
                                     level,
                                     block_sums_device + offset);

        offset += z_blocks_max;
    }

    error_check(cudaDeviceSynchronize())
    error_check(cudaMemcpy(block_sums_host, block_sums_device, z_blocks_max * num_levels * sizeof(int), cudaMemcpyDeviceToHost) )

    int counter = 0;
    offset = 0;

    for(int level = tree_access.level_min(); level <= tree_access.level_max(); ++level) {
        ne_count[level+1] = counter;

        for(int i = 0; i < z_blocks_max; ++i) {
            counter += block_sums_host[offset + i];
        }
        offset += z_blocks_max;
    }

    ne_count.back() = counter;
    ne_rows_gpu.initialize(NULL, counter);

    for(int level = (tree_access.level_min() + 1); level <= (tree_access.level_max() + 1); ++level) {

        int ne_sz = ne_count[level+1] - ne_count[level];

        if( ne_sz == 0 ) {
            continue;
        }

        int z_blocks = (tree_access.z_num(level - 1) + blockSize_z - 1) / blockSize_z;

        dim3 grid_dim(z_blocks, 1, 1);
        dim3 block_dim(blockSize_z, blockSize_x, 1);

        _fill_ne_rows_tree_cuda<<< grid_dim, block_dim >>>
                                       (tree_access.get_level_xz_vec_ptr(),
                                        tree_access.get_xz_end_vec_ptr(),
                                        tree_access.z_num(level-1),
                                        tree_access.x_num(level-1),
                                        level-1,
                                        ne_sz,
                                        ne_count[level],
                                        ne_rows_gpu.get());
    }

    error_check(cudaFree(block_sums_device))
}


void compute_ne_rows_tree(GPUAccessHelper& tree_access, VectorData<int>& ne_counter, VectorData<int>& ne_rows) {
    ne_counter.resize(tree_access.level_max() + 3);
    ne_counter[0] = 0;

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
void downsample_avg(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu, int* ne_rows,VectorData<int>& ne_offset) {

    /// assumes input_gpu, tree_data_gpu and ne_rows are already on the device

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        size_t ne_sz = ne_offset[level+1] - ne_offset[level];
        size_t offset = ne_offset[level];

        if( ne_sz == 0 ) {
            continue;
        }

        dim3 threads_l(128, 1, 1);
        dim3 blocks_l(ne_sz, 1, 1);

        if(level == access.level_max()){

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
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;
    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter, ne_rows_gpu);

    error_check( cudaDeviceSynchronize() )

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
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;
    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter, ne_rows_gpu);

    input_gpu.copyH2D();

    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);

    tree_data_gpu.copyD2H();
}

/// instantiate templates
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint8_t>&, VectorData<float>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint64_t>&, VectorData<float>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&);
template void downsample_avg(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&);

template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint8_t>&, VectorData<float>&);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint64_t>&, VectorData<float>&);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&);
template void downsample_avg_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&);

template void compute_ne_rows_tree_cuda<8, 32>(GPUAccessHelper&, VectorData<int>&, ScopedCudaMemHandler<int*, JUST_ALLOC>&);
template void compute_ne_rows_tree_cuda<16, 32>(GPUAccessHelper&, VectorData<int>&, ScopedCudaMemHandler<int*, JUST_ALLOC>&);
template void compute_ne_rows_tree_cuda<32, 32>(GPUAccessHelper&, VectorData<int>&, ScopedCudaMemHandler<int*, JUST_ALLOC>&);
