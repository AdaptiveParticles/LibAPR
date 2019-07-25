//
// Created by cheesema on 05.04.18.
//

#include "APRDownsampleGPU.hpp"

//__device__ void get_row_begin_end(std::size_t* index_begin,
//                                  std::size_t* index_end,
//                                  std::size_t xz_start,
//                                  const uint64_t* xz_end_vec){
//
//    *index_end = (xz_end_vec[xz_start]);
//
//    if (xz_start == 0) {
//        *index_begin = 0;
//    } else {
//        *index_begin =(xz_end_vec[xz_start-1]);
//    }
//}


template<typename inputType, typename outputType>
__global__ void down_sample_avg(const uint64_t* level_xz_vec,
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

    const int x_index = (2 * blockIdx.x + threadIdx.x/64);
    const int z_index = (2 * blockIdx.z + ((threadIdx.x)/32)%2);

    const int block = threadIdx.x/32;
    const int local_th = (threadIdx.x%32);


    __shared__ size_t global_index_begin_0_s[4];
    __shared__ size_t global_index_end_0_s[4];

    size_t global_index_begin_p;
    size_t global_index_end_p;

    //remove these with registers
    //__shared__ float f_cache[5][32];
    //__shared__ int y_cache[5][32];


    int current_y = -1;
    int current_y_p = -1;

    if( (x_index >= x_num) || (z_index >= z_num) ){

         return; //out of bounds
    } else {

        if(threadIdx.x==0){

        }
        //get_row_begin_end(&global_index_begin_0, &global_index_end_0, x_index + z_index*x_num + level_xz_vec[level], xz_end_vec);
        if((local_th==0) ) {
            size_t xz_start_s = x_index + z_index * x_num + level_xz_vec[level];
            global_index_begin_0_s[block] = xz_end_vec[xz_start_s - 1];
            global_index_end_0_s[block] = xz_end_vec[xz_start_s];
        }
    }
    __syncthreads();

    if(global_index_begin_0_s[0] == global_index_end_0_s[0]){
        return;
    }

    size_t global_index_begin_0 = global_index_begin_0_s[block];
    size_t global_index_end_0 = global_index_end_0_s[block];




    //keep these
    __shared__ float parent_cache[8][16];

    float current_val = 0;

    parent_cache[2*block][local_th/2] = 0;
    parent_cache[2*block+1][local_th/2] = 0;

    float scale_factor_xz = (((2*x_num_parent != x_num) && blockIdx.x==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && blockIdx.z==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    float scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }


    //get_row_begin_end(&global_index_begin_p, &global_index_end_p, blockIdx.x + blockIdx.z*x_num_parent + level_xz_vec_tree[level-1], xz_end_vec_tree);
    size_t xz_start = blockIdx.x + blockIdx.z*x_num_parent + level_xz_vec_tree[level-1];
    global_index_begin_p = xz_end_vec_tree[xz_start - 1];
    global_index_end_p = xz_end_vec_tree[xz_start];

    //initialize (i=0)
    if ((global_index_begin_0 + local_th) < global_index_end_0) {
        current_val = input_particles[global_index_begin_0 + local_th];
        current_y =  y_vec[global_index_begin_0 + local_th];
    }


    if (block == 3) {

        if (( global_index_begin_p + local_th) < global_index_end_p) {

            current_y_p = y_vec_tree[global_index_begin_p + local_th];

        }

    }

    uint16_t sparse_block = 0;
    int sparse_block_p = 0;
    const uint16_t number_y_chunk = (y_num+31)/32;

    for (int y_block = 0; y_block < number_y_chunk; ++y_block) {

        __syncthreads();
        //value less then current chunk then update.
        if (current_y < y_block * 32) {
            sparse_block++;
            if ((sparse_block * 32 + global_index_begin_0 + local_th) < global_index_end_0) {
                current_val = input_particles[sparse_block * 32 + global_index_begin_0 + local_th];

                current_y = y_vec[sparse_block * 32 + global_index_begin_0 + local_th];
            }
        }

        //current_y = y_cache[block][local_th];
        __syncthreads();

        //update the down-sampling caches
        if ((current_y < (y_block + 1) * 32) && (current_y >= (y_block) * 32)) {

            parent_cache[2*block+current_y%2][(current_y/2) % 16] = (1.0f/8.0f)*current_val;
            //parent_cache[2*block+current_y%2][(current_y/2) % 16] = 1;

        }

        __syncthreads();
        //fetch the parent particle data
        if (block == 3) {
            if (current_y_p < ((y_block * 32)/2)) {
                sparse_block_p++;


                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {

                    current_y_p = y_vec_tree[sparse_block_p * 32 + global_index_begin_p + local_th];

                }

            }


        }
        __syncthreads();

        if(block ==3) {
            //output

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
        parent_cache[2*block][local_th/2] = 0;
        parent_cache[2*block+1][local_th/2] = 0;
    }
}

template<typename inputType, typename outputType>
__global__ void down_sample_avg_interior(const uint64_t* level_xz_vec,
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

    int x_index = (2 * blockIdx.x + threadIdx.x/64);
    int z_index = (2 * blockIdx.z + ((threadIdx.x)/32)%2);



    const int block = threadIdx.x/32;
    const int local_th = (threadIdx.x%32);

    //Particles
    std::size_t global_index_begin_0;
    std::size_t global_index_end_0;

    //Parent Tree Particle Cells
    std::size_t global_index_begin_p;
    std::size_t global_index_end_p;

    //Interior Tree Particle Cells
    std::size_t global_index_begin_t;
    std::size_t global_index_end_t;



    int current_y=-1;
    int current_y_p=-1;
    int current_y_t=-1;
    float current_val=0;
    float current_val_t = 0;

    if((x_index >= x_num) || (z_index >= z_num) ){

        global_index_begin_0 = 1;
        global_index_end_0 = 0;

        global_index_begin_t = 1;
        global_index_end_t = 0;

        // return; //out of bounds
    } else {
        //get_row_begin_end(&global_index_begin_t, &global_index_end_t, x_index + z_index*x_num + level_xz_vec_tree[level], xz_end_vec_tree);
        //get_row_begin_end(&global_index_begin_0, &global_index_end_0, x_index + z_index*x_num + level_xz_vec[level], xz_end_vec);
        size_t xz_start = x_index + z_index * x_num + level_xz_vec_tree[level];
        global_index_begin_t = xz_end_vec_tree[xz_start - 1];
        global_index_end_t = xz_end_vec_tree[xz_start];

        xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin_0 = xz_end_vec[xz_start - 1];
        global_index_end_0 = xz_end_vec[xz_start];

//        if ((global_index_begin_t < global_index_end_t) && (local_th == 0)) {
//            printf("level %d, tree: [%d, %d), apr: [%d, %d)\n", (int) level, (int) global_index_begin_t,
//                   (int) global_index_end_t, (int) global_index_begin_0, (int) global_index_end_0);
//        }
    }

    //get_row_begin_end(&global_index_begin_p, &global_index_end_p, blockIdx.x + blockIdx.z*x_num_parent + level_xz_vec_tree[level-1], xz_end_vec_tree);
    size_t xz_start = blockIdx.x + blockIdx.z*x_num_parent + level_xz_vec_tree[level-1];
    global_index_begin_p = xz_end_vec_tree[xz_start - 1];
    global_index_end_p = xz_end_vec_tree[xz_start];

    //initialize (i=0)
    if ((global_index_begin_0 + local_th) < global_index_end_0) {

        current_y = y_vec[global_index_begin_0 + local_th];
        current_val = input_particles[global_index_begin_0 + local_th];

    }

    //tree interior
    if ((global_index_begin_t + local_th) < global_index_end_t) {

        current_y_t = y_vec_tree[global_index_begin_t + local_th];
        current_val_t = particle_data_output[global_index_begin_t + local_th];
    }

    if((global_index_begin_0 == global_index_end_0) && (global_index_begin_t == global_index_end_t)){
        return;
    }


    //shared memory caches

    __shared__ float parent_cache[8][16];


    parent_cache[2*block][local_th/2]=0;
    parent_cache[2*block+1][local_th/2]=0;

    float scale_factor_xz = (((2*x_num_parent != x_num) && blockIdx.x==(x_num_parent-1) ) + ((2*z_num_parent != z_num) && blockIdx.z==(z_num_parent-1) ))*2;

    if(scale_factor_xz == 0){
        scale_factor_xz = 1;
    }

    float scale_factor_yxz = scale_factor_xz;

    if((2*y_num_parent != y_num)){
        scale_factor_yxz = scale_factor_xz*2;
    }



    if (block == 3) {

        if (( global_index_begin_p + local_th) < global_index_end_p) {

            current_y_p = y_vec_tree[global_index_begin_p + local_th];

        }
    }

    int sparse_block = 0;
    int sparse_block_p = 0;
    int sparse_block_t = 0;

    const int number_y_chunk = (y_num+31)/32;

    for (int y_block = 0; y_block < (number_y_chunk); ++y_block) {

        __syncthreads();
        //value less then current chunk then update.
        if (current_y < (y_block * 32)) {
            sparse_block++;
            if ((sparse_block * 32 + global_index_begin_0 + local_th) < global_index_end_0) {

                current_val = input_particles[sparse_block * 32 + global_index_begin_0 + local_th];
                current_y = y_vec[sparse_block * 32 + global_index_begin_0 + local_th];
            }
        }

        //interior tree update
        if (current_y_t < (y_block * 32)) {
            sparse_block_t++;
            if ((sparse_block_t * 32 + global_index_begin_t + local_th) < global_index_end_t) {

                current_val_t = particle_data_output[sparse_block_t * 32 + global_index_begin_t + local_th];
                current_y_t = y_vec_tree[sparse_block_t * 32 + global_index_begin_t + local_th];
            }
        }
        // current_y_t = y_cache_t[block][local_th];

        __syncthreads();
        //update the down-sampling caches
        if ((current_y < (y_block + 1) * 32) && (current_y >= (y_block) * 32)) {

            parent_cache[2*block+current_y%2][(current_y/2) % 16] = (1.0/8.0f)*current_val;
            //parent_cache[2*block+current_y%2][(current_y/2) % 16] = 1;

        }
        __syncthreads();



        //now the interior tree nodes
        if ((current_y_t < (y_block + 1) * 32) && (current_y_t >= (y_block) * 32)) {

            parent_cache[2*block + current_y_t%2][(current_y_t/2) % 16] = (1.0/8.0f)*current_val_t;
            //parent_cache[2*block+current_y_t%2][(current_y_t/2) % 16] = 1;
            //parent_cache[0][(current_y_t/2) % 16] = current_y_t/2;
        }
        __syncthreads();


        if (block == 3) {

            if (current_y_p < ((y_block * 32)/2)) {
                sparse_block_p++;

                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {

                    //y_cache[4][local_th] = particle_y_child[sparse_block_p * 32 + global_index_begin_p + local_th];
                    current_y_p = y_vec_tree[sparse_block_p * 32 + global_index_begin_p + local_th];

                }
            }
        }

        __syncthreads();

        //local_sum
        if(block ==3) {
            //output
            //current_y_p = y_cache[4][local_th];
            current_y_p = y_vec_tree[sparse_block_p * 32 + global_index_begin_p + local_th];

            if (current_y_p < ((y_block+1) * 32)/2 && current_y_p >= ((y_block) * 32)/2) {
                if ((sparse_block_p * 32 + global_index_begin_p + local_th) < global_index_end_p) {

                    if (current_y_p == (y_num_parent - 1)) {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_yxz * (parent_cache[0][current_y_p % 16] +
                                                    parent_cache[1][current_y_p % 16] +
                                                    parent_cache[2][current_y_p % 16] +
                                                    parent_cache[3][current_y_p % 16] +
                                                    parent_cache[4][current_y_p % 16] +
                                                    parent_cache[5][current_y_p % 16] +
                                                    parent_cache[6][current_y_p % 16] +
                                                    parent_cache[7][current_y_p % 16]);


                    } else {
                        particle_data_output[sparse_block_p * 32 + global_index_begin_p + local_th] =
                                scale_factor_xz * ( parent_cache[0][current_y_p % 16] +
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

        parent_cache[2*block][local_th/2] = 0;
        parent_cache[2*block+1][local_th/2] = 0;

    }
}

template<typename inputType, typename treeType>
void downsample_avg_init_wrapper(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input, std::vector<treeType>& tree_data) {

    tree_data.resize(tree_access.total_number_particles());

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());

    /// copy the input to the GPU
    input_gpu.copyH2D();

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        dim3 threads_l(128, 1, 1);

        int x_blocks = (access.x_num(level) + 2 - 1) / 2;
        int z_blocks = (access.z_num(level) + 2 - 1) / 2;

        dim3 blocks_l(x_blocks, 1, z_blocks);

        if(level==access.level_max()) {

            down_sample_avg << < blocks_l, threads_l >> >
                                               (access.get_level_xz_vec_ptr(),
                                                       access.get_xz_end_vec_ptr(),
                                                       access.get_y_vec_ptr(),
                                                       input_gpu.get(),
                                                       tree_access.get_level_xz_vec_ptr(),
                                                       tree_access.get_xz_end_vec_ptr(),
                                                       tree_access.get_y_vec_ptr(),
                                                       tree_data_gpu.get(),
                                                       access.z_num(level),
                                                       access.x_num(level),
                                                       access.y_num(level),
                                                       tree_access.z_num(level-1),
                                                       tree_access.x_num(level-1),
                                                       tree_access.y_num(level-1),
                                                       level);


        } else {

            down_sample_avg_interior<< < blocks_l, threads_l >> >
                                                       (access.get_level_xz_vec_ptr(),
                                                        access.get_xz_end_vec_ptr(),
                                                        access.get_y_vec_ptr(),
                                                        input_gpu.get(),
                                                        tree_access.get_level_xz_vec_ptr(),
                                                        tree_access.get_xz_end_vec_ptr(),
                                                        tree_access.get_y_vec_ptr(),
                                                        tree_data_gpu.get(),
                                                        access.z_num(level),
                                                        access.x_num(level),
                                                        access.y_num(level),
                                                        tree_access.z_num(level-1),
                                                        tree_access.x_num(level-1),
                                                        tree_access.y_num(level-1),
                                                        level);
        }
        //cudaDeviceSynchronize();
    }

    /// transfer the results back to the host
    tree_data_gpu.copyD2H();
}



template<typename inputType, typename treeType>
void downsample_avg_wrapper(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, treeType* tree_data_gpu) {

    /// assumes input_gpu and tree_data_gpu are already on the device

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        dim3 threads_l(128, 1, 1);

        int x_blocks = (access.x_num(level) + 2 - 1) / 2;
        int z_blocks = (access.z_num(level) + 2 - 1) / 2;

        dim3 blocks_l(x_blocks, 1, z_blocks);

        if(level==access.level_max()) {

            down_sample_avg << < blocks_l, threads_l >> >
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

            down_sample_avg_interior<< < blocks_l, threads_l >> >
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
        //cudaDeviceSynchronize();
    }
}
