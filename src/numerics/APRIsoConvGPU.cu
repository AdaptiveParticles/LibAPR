//
// Created by cheesema on 09.04.18.
//

#include "APRIsoConvGPU.hpp"

#define LOCALPATCHCONV333(particle_output,index,z,x,y,neighbour_sum)\
neighbour_sum=0;\
if (not_ghost) {\
    for (int q = 0; q < 3; ++q) {\
neighbour_sum  +=  stencil[q*9]*local_patch[z + q - 1][x + 0 - 1][(y+N-1)%N]\
                 + stencil[q*9+1]*local_patch[z + q - 1][x + 0 - 1][(y+N)%N]\
                 + stencil[q*9+2]*local_patch[z + q - 1][x + 0 - 1][(y+N+1)%N]\
                 + stencil[q*9+3]*local_patch[z + q - 1][x + 1 - 1][(y+N-1)%N]\
                 + stencil[q*9+4]*local_patch[z + q - 1][x + 1 - 1][(y+N)%N]\
                 + stencil[q*9+5]*local_patch[z + q - 1][x + 1 - 1][(y+N+1)%N]\
                 + stencil[q*9+6]*local_patch[z + q - 1][x + 2 - 1][(y+N-1)%N]\
                 + stencil[q*9+7]*local_patch[z + q - 1][x + 2 - 1][(y+N)%N]\
                 + stencil[q*9+8]*local_patch[z + q - 1][x + 2 - 1][(y+N+1)%N];\
    }\
    particle_output[index] = neighbour_sum;\
}\


//
//neighbour_sum += stencil[q*9]*local_patch[z + q - 1][x + 0 - 1][(y+N)%N]\
//                 + stencil[q*9+1]*local_patch[z + q - 1][x + 0 - 1][(y+N-1)%N]\
//                 + stencil[q*9+2]*local_patch[z + q - 1][x + 0 - 1][(y+N+1)%N]\
//                 + stencil[q*9+3]*local_patch[z + q - 1][x + 1 - 1][(y+N)%N]\
//                 + stencil[q*9+4]*local_patch[z + q - 1][x + 1 - 1][(y+N-1)%N]\
//                 + stencil[q*9+5]*local_patch[z + q - 1][x + 1 - 1][(y+N+1)%N]\
//                 + stencil[q*9+6]*local_patch[z + q - 1][x + 2 - 1][(y+N)%N]\
//                 + stencil[q*9+7]*local_patch[z + q - 1][x + 2 - 1][(y+N-1)%N]\
//                 + stencil[q*9+8]*local_patch[z + q - 1][x + 2 - 1][(y+N+1)%N];\

#define LOCALPATCHCONV555(particle_output,index,z,x,y,neighbour_sum)\
neighbour_sum=0;\
if (not_ghost) {\
for (int q = 0; q < 5; ++q) {\
            neighbour_sum +=\
                stencil[q*25+0]*local_patch[z + q - 2][x + 0 - 2][(y+N-2)%N]\
                 + stencil[q*25+1]*local_patch[z + q - 2][x + 0 - 2][(y+N-1)%N]\
                 + stencil[q*25+2]*local_patch[z + q - 2][x + 0 - 2][(y+N)%N]\
                 + stencil[q*25+3]*local_patch[z + q - 2][x + 0 - 2][(y+N+1)%N]\
                 + stencil[q*25+4]*local_patch[z + q - 2][x + 0 - 2][(y+N+2)%N]\
                 + stencil[q*25+5]*local_patch[z + q - 2][x + 1 - 2][(y+N-2)%N]\
                 + stencil[q*25+6]*local_patch[z + q - 2][x + 1 - 2][(y+N-1)%N]\
                 + stencil[q*25+7]*local_patch[z + q - 2][x + 1 - 2][(y+N)%N]\
                 + stencil[q*25+8]*local_patch[z + q - 2][x + 1 - 2][(y+N+1)%N]\
                 + stencil[q*25+9]*local_patch[z + q - 2][x + 1 - 2][(y+N+2)%N]\
                + stencil[q*25+10]*local_patch[z + q - 2][x + 2 - 2][(y+N-2)%N]\
                 + stencil[q*25+11]*local_patch[z + q - 2][x + 2 - 2][(y+N-1)%N]\
                 + stencil[q*25+12]*local_patch[z + q - 2][x + 2 - 2][(y+N)%N]\
                 + stencil[q*25+13]*local_patch[z + q - 2][x + 2 - 2][(y+N+1)%N]\
                 + stencil[q*25+14]*local_patch[z + q - 2][x + 2 - 2][(y+N+2)%N]\
                + stencil[q*25+15]*local_patch[z + q - 2][x + 3 - 2][(y+N-2)%N]\
                 + stencil[q*25+16]*local_patch[z + q - 2][x + 3 - 2][(y+N-1)%N]\
                 + stencil[q*25+17]*local_patch[z + q - 2][x + 3 - 2][(y+N)%N]\
                 + stencil[q*25+18]*local_patch[z + q - 2][x + 3 - 2][(y+N+1)%N]\
                 + stencil[q*25+19]*local_patch[z + q - 2][x + 3 - 2][(y+N+2)%N]\
                + stencil[q*25+20]*local_patch[z + q - 2][x + 4 - 2][(y+N-2)%N]\
                 + stencil[q*25+21]*local_patch[z + q - 2][x + 4 - 2][(y+N-1)%N]\
                 + stencil[q*25+22]*local_patch[z + q - 2][x + 4 - 2][(y+N)%N]\
                 + stencil[q*25+23]*local_patch[z + q - 2][x + 4 - 2][(y+N+1)%N]\
                 + stencil[q*25+24]*local_patch[z + q - 2][x + 4 - 2][(y+N+2)%N];\
}\
particle_output[index] = neighbour_sum;\
}\


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_333_wrapper(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil) {

    assert(stencil.size() == 27);

    timings ret;
    APRTimer timer(false);

    timer.start_timer("init output");
    output.init(input);
    timer.stop_timer();

    timer.start_timer("transfer H2D");

    /// allocate GPU memory
    ScopedCudaMemHandler<PixelData<inputType>, JUST_ALLOC> input_gpu(input);
    ScopedCudaMemHandler<PixelData<outputType>, JUST_ALLOC> output_gpu(output);
    ScopedCudaMemHandler<PixelData<stencilType>, JUST_ALLOC> stencil_gpu(stencil);

    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    cudaDeviceSynchronize();

    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    timer.start_timer("run kernel");

    dim3 threads_l(10, 1, 10);

    int x_blocks = (input.x_num + 8 - 1) / 8;
    int z_blocks = (input.z_num + 8 - 1) / 8;

    dim3 blocks_l(x_blocks, 1, z_blocks);

    conv_pixel_333<<< blocks_l, threads_l >>>(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.z_num, input.x_num, input.y_num);

    cudaDeviceSynchronize();
    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


template<typename inputType, typename outputType, typename stencilType>
timings convolve_pixel_555_wrapper(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil) {

    assert(stencil.size() == 125);

    timings ret;
    APRTimer timer(false);

    timer.start_timer("init output");
    output.init(input);
    timer.stop_timer();

    timer.start_timer("transfer H2D");

    /// allocate GPU memory
    ScopedCudaMemHandler<PixelData<inputType>, JUST_ALLOC> input_gpu(input);
    ScopedCudaMemHandler<PixelData<outputType>, JUST_ALLOC> output_gpu(output);
    ScopedCudaMemHandler<PixelData<stencilType>, JUST_ALLOC> stencil_gpu(stencil);

    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    cudaDeviceSynchronize();

    timer.stop_timer();

    ret.transfer_H2D = timer.timings.back();

    timer.start_timer("run kernels");

    dim3 threads_l(12, 1, 12);

    int x_blocks = (input.x_num + 8 - 1) / 8;
    int z_blocks = (input.z_num + 8 - 1) / 8;

    dim3 blocks_l(x_blocks, 1, z_blocks);

    conv_pixel_555<<< blocks_l, threads_l >>>(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.z_num, input.x_num, input.y_num);

    cudaDeviceSynchronize();
    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333_wrapper(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
                                    std::vector<outputType>& output, std::vector<stencilType>& stencil, std::vector<treeType>& tree_data){
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel
     *
     *  conv_stencil needs to have 27 entries
     */

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 27);

    timings ret;
    APRTimer timer(false);

    timer.start_timer("init arrays");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();

    timer.start_timer("transfer H2D");

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    size_t max_num_blocks = ((access.x_num(access.level_max()) + 8 - 1) / 8) * ((access.z_num(access.level_max()) + 8 - 1) / 8);
    ScopedCudaMemHandler<bool*, JUST_ALLOC> blocks_empty(NULL, max_num_blocks);

    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    cudaDeviceSynchronize();
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("fill tree");
    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    cudaDeviceSynchronize();
    timer.stop_timer();

    ret.fill_tree = timer.timings.back();

    timer.start_timer("run kernels");

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        int x_blocks = (access.x_num(level) + 8 - 1) / 8;
        int z_blocks = (access.z_num(level) + 8 - 1) / 8;

        dim3 blocks_l(x_blocks, 1, z_blocks);
        dim3 threads(1,1,1);

        check_blocks<<<blocks_l, threads>>>(access.get_level_xz_vec_ptr(),
                                            access.get_xz_end_vec_ptr(),
                                            blocks_empty.get(),
                                            8 /* block size */,
                                            level,
                                            access.x_num(level));

        dim3 threads_l(10, 1, 10);

        if (level == access.level_min()) {
            conv_min_333 << < blocks_l, threads_l >> >( access.get_level_xz_vec_ptr(),
                                                        access.get_xz_end_vec_ptr(),
                                                        access.get_y_vec_ptr(),
                                                        input_gpu.get(),
                                                        output_gpu.get(),
                                                        stencil_gpu.get(),
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
                                                        level,
                                                        blocks_empty.get() );

        } else if (level == access.level_max()) {
            conv_max_333 << < blocks_l, threads_l >> >( access.get_level_xz_vec_ptr(),
                                                        access.get_xz_end_vec_ptr(),
                                                        access.get_y_vec_ptr(),
                                                        input_gpu.get(),
                                                        output_gpu.get(),
                                                        stencil_gpu.get(),
                                                        access.z_num(level),
                                                        access.x_num(level),
                                                        access.y_num(level),
                                                        tree_access.z_num(level-1),
                                                        tree_access.x_num(level-1),
                                                        tree_access.y_num(level-1),
                                                        level,
                                                        blocks_empty.get() );

        } else {
            conv_interior_333 << < blocks_l, threads_l >> >(access.get_level_xz_vec_ptr(),
                                                            access.get_xz_end_vec_ptr(),
                                                            access.get_y_vec_ptr(),
                                                            input_gpu.get(),
                                                            output_gpu.get(),
                                                            stencil_gpu.get(),
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
                                                            level,
                                                            blocks_empty.get() );
        }
        cudaDeviceSynchronize();
    }

    timer.stop_timer();

    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555_wrapper(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
                                    std::vector<outputType>& output, std::vector<stencilType>& stencil, std::vector<treeType>& tree_data){
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 5x5x5 kernel
     *
     *  conv_stencil needs to have 125 entries
     */

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 125);

    timings ret;
    APRTimer timer(false);

    timer.start_timer("init arrays");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();

    timer.start_timer("transfer H2D");

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    size_t max_num_blocks = ((access.x_num(access.level_max()) + 8 - 1) / 8) * ((access.z_num(access.level_max()) + 8 - 1) / 8);
    ScopedCudaMemHandler<bool*, JUST_ALLOC> blocks_empty(NULL, max_num_blocks);

    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    cudaDeviceSynchronize();

    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("fill tree");
    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    cudaDeviceSynchronize();
    timer.stop_timer();

    ret.fill_tree = timer.timings.back();

    timer.start_timer("run kernels");

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        int x_blocks = (access.x_num(level) + 8 - 1) / 8;
        int z_blocks = (access.z_num(level) + 8 - 1) / 8;

        dim3 blocks_l(x_blocks, 1, z_blocks);
        dim3 threads(1,1,1);

        check_blocks<<<blocks_l, threads>>>(access.get_level_xz_vec_ptr(),
                                            access.get_xz_end_vec_ptr(),
                                            blocks_empty.get(),
                                            8 /* block size */,
                                            level,
                                            access.x_num(level));

        dim3 threads_l(12, 1, 12);

        if (level == access.level_min()) {
            conv_min_555 << < blocks_l, threads_l >> >( access.get_level_xz_vec_ptr(),
                    access.get_xz_end_vec_ptr(),
                    access.get_y_vec_ptr(),
                    input_gpu.get(),
                    output_gpu.get(),
                    stencil_gpu.get(),
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
                    level,
                    blocks_empty.get());

        } else if (level == access.level_max()) {
            conv_max_555 << < blocks_l, threads_l >> >( access.get_level_xz_vec_ptr(),
                    access.get_xz_end_vec_ptr(),
                    access.get_y_vec_ptr(),
                    input_gpu.get(),
                    output_gpu.get(),
                    stencil_gpu.get(),
                    access.z_num(level),
                    access.x_num(level),
                    access.y_num(level),
                    tree_access.z_num(level-1),
                    tree_access.x_num(level-1),
                    tree_access.y_num(level-1),
                    level,
                    blocks_empty.get());

        } else {
            conv_interior_555 << < blocks_l, threads_l >> >(access.get_level_xz_vec_ptr(),
                    access.get_xz_end_vec_ptr(),
                    access.get_y_vec_ptr(),
                    input_gpu.get(),
                    output_gpu.get(),
                    stencil_gpu.get(),
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
                    level,
                    blocks_empty.get());
        }
        cudaDeviceSynchronize();
    }

    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}



__global__ void check_blocks(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             bool* blocks_empty,
                             const int block_size,
                             const int level,
                             const int x_num) {

    size_t x_start = block_size * blockIdx.x;
    size_t z_start = block_size * blockIdx.z;

    size_t xz_start, global_index_begin, global_index_end;

    for(size_t iz = 0; iz < block_size; ++iz) {
        for(size_t ix = 0; ix < block_size; ++ix) {

            xz_start = (x_start + ix) + (z_start + iz) * x_num + level_xz_vec[level];
            global_index_begin = xz_end_vec[xz_start - 1];
            global_index_end = xz_end_vec[xz_start];

            if(global_index_begin < global_index_end) {
                blocks_empty[blockIdx.z * gridDim.x + blockIdx.x] = false;
                return;
            }
        }
    }
    printf("block %d empty!\n", (blockIdx.z * gridDim.x + blockIdx.x));
    blocks_empty[blockIdx.z * gridDim.x + blockIdx.x] = true;
}


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             const uint16_t* y_vec,
                             const inputType* input_particles,
                             outputType* particle_data_output,
                             const stencilType* stencil,
                             const int z_num,
                             const int x_num,
                             const int y_num,
                             const int z_num_parent,
                             const int x_num_parent,
                             const int y_num_parent,
                             const int level,
                             const bool* blocks_empty){

    if(blocks_empty[blockIdx.z * gridDim.x + blockIdx.x]) {
        return;
    }

    // This is block wise shared memory this is assuming an 8*8 block with pad()

    bool not_ghost = false;

    if ((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)) {
        not_ghost = true;
    }

    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);

    const unsigned int N = 4;

    __shared__
    stencilType local_patch[10][10][6];

    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        return; //out of bounds
    }


    std::size_t global_index_begin;
    std::size_t global_index_end;

    std::size_t global_index_begin_p;
    std::size_t global_index_end_p;

    // current level
    std::size_t xz_start = x_index + z_index * x_num + level_xz_vec[level];

    global_index_begin = xz_end_vec[xz_start - 1];
    global_index_end = xz_end_vec[xz_start];

    std::size_t particle_index_l = global_index_begin;
    std::uint16_t y_l = y_vec[particle_index_l];
    inputType f_l = input_particles[particle_index_l];

    // parent level, level - 1, one resolution lower (coarser)
    int x_index_p = (x_index) / 2;
    int z_index_p = (z_index) / 2;

    xz_start = x_index_p + z_index_p * x_num_parent + level_xz_vec[level - 1];

    global_index_begin_p = xz_end_vec[xz_start - 1];
    global_index_end_p = xz_end_vec[xz_start];

    //parent level variables
    std::size_t particle_index_p = global_index_begin_p;
    std::uint16_t y_p = y_vec[particle_index_p];
    inputType f_p = input_particles[particle_index_p];


    if (global_index_begin_p == global_index_end_p) {
        y_p = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin == global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }

    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)

    const int filter_offset = 1;

    __shared__
    std::uint16_t y_update_flag[10][10][2];
    __shared__
    std::uint16_t y_update_index[10][10][2];

    y_update_flag[threadIdx.z][threadIdx.x][0] = 0;
    y_update_flag[threadIdx.z][threadIdx.x][1] = 0;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T
        __syncthreads();
        //Check if its time to update the parent level

        if (j == (2 * y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N] = f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j + 1) % N] = f_p;
        }


        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][j % N] = f_l; //initial update
            y_update_flag[threadIdx.z][threadIdx.x][j % 2] = 1;
            y_update_index[threadIdx.z][threadIdx.x][j % 2] = particle_index_l - global_index_begin;
        } else {
            y_update_flag[threadIdx.z][threadIdx.x][j % 2] = 0;
        }

        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < global_index_end)) {
            particle_index_l++;
            y_l = y_vec[particle_index_l];
            f_l = input_particles[particle_index_l];
        }

        //parent update loop
        if ((2 * y_p <= j) && ((particle_index_p + 1) < global_index_end_p)) {
            particle_index_p++;
            y_p = y_vec[particle_index_p];
            f_p = input_particles[particle_index_p];
        }

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[threadIdx.z][threadIdx.x][(j - filter_offset + 2) % 2] == 1) {

            float neighbour_sum = 0;
            LOCALPATCHCONV333(particle_data_output, global_index_begin +
                              y_update_index[threadIdx.z][threadIdx.x][(j + 2 - filter_offset) % 2],
                              threadIdx.z, threadIdx.x, j - 1, neighbour_sum);

        }
    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[threadIdx.z][threadIdx.x][(y_num - 1) % 2] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1,
                          neighbour_sum);


    }


}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333(const uint64_t* level_xz_vec,
                                  const uint64_t* xz_end_vec,
                                  const uint16_t* y_vec,
                                  const inputType* input_particles,
                                  outputType* particle_data_output,
                                  const stencilType* stencil,
                                  const uint64_t* level_xz_vec_tree,
                                  const uint64_t* xz_end_vec_tree,
                                  const uint16_t* y_vec_tree,
                                  const treeType* tree_data,
                                  const int z_num,
                                  const int x_num,
                                  const int y_num,
                                  const int z_num_parent,
                                  const int x_num_parent,
                                  const int y_num_parent,
                                  const int level,
                                  const bool* blocks_empty) {

    if(blocks_empty[blockIdx.z * gridDim.x + blockIdx.x]) {
        return;
    }

    const unsigned int N = 4;

    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);


    bool not_ghost = false;

    if ((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)) {
        not_ghost = true;
    }

    __shared__
    stencilType local_patch[10][10][6]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    if ((x_index >= x_num) || (x_index < 0)) {
        //set the whole buffer to the boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        //set the whole buffer to the zero boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    int x_index_p = (8 * blockIdx.x + threadIdx.x - 1) / 2;
    int z_index_p = (8 * blockIdx.z + threadIdx.z - 1) / 2;


    std::size_t global_index_begin;
    std::size_t global_index_end;

    std::size_t global_index_begin_p;
    std::size_t global_index_end_p;

    // current level
    std::size_t xz_start = level_xz_vec[level] + (x_index) + (z_index) * x_num;
    global_index_begin = xz_end_vec[xz_start - 1];
    global_index_end = xz_end_vec[xz_start];

    std::size_t particle_index_l = global_index_begin;
    std::uint16_t y_l = y_vec[particle_index_l];
    //inputType f_l = input_particles[particle_index_l];

    // parent level, level - 1, one resolution lower (coarser)
    xz_start = level_xz_vec[level - 1] + (x_index_p) + (z_index_p) * x_num_parent;
    global_index_begin_p = xz_end_vec[xz_start - 1];
    global_index_end_p = xz_end_vec[xz_start];

    //parent level variables
    std::size_t particle_index_p = global_index_begin_p;
    std::uint16_t y_p = y_vec[particle_index_p];
    inputType f_p = input_particles[particle_index_p];


    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    xz_start = level_xz_vec_tree[level] + (x_index) + (z_index) * x_num;

    std::size_t global_index_begin_t = xz_end_vec_tree[xz_start - 1];
    std::size_t global_index_end_t = xz_end_vec_tree[xz_start];

    std::size_t particle_index_t = global_index_begin_t;
    std::uint16_t y_t = y_vec_tree[particle_index_t];
    treeType f_t = tree_data[particle_index_t];


    if (global_index_begin_t == global_index_end_t) {
        y_t = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin_p == global_index_end_p) {
        y_p = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin == global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }

    __shared__
    std::uint16_t y_update_flag[10][10][2];
    __shared__
    std::uint16_t y_update_index[10][10][2];

    __shared__
    std::uint16_t f_l[10][10];

    f_l[threadIdx.z][threadIdx.x] = input_particles[particle_index_l];

    y_update_flag[threadIdx.z][threadIdx.x][0] = 0;
    y_update_flag[threadIdx.z][threadIdx.x][1] = 0;

    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)

    const int filter_offset = 1;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        //Check if its time to update the parent level
        if (j == (2 * y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N] = f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j + 1) % N] = f_p;
        }

        //Check if its time to update child level
        if (j == y_t) {
            local_patch[threadIdx.z][threadIdx.x][y_t % N] = f_t; //initial update
        }

        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N] = f_l[threadIdx.z][threadIdx.x]; //initial update
            y_update_flag[threadIdx.z][threadIdx.x][j % 2] = 1;
            y_update_index[threadIdx.z][threadIdx.x][j % 2] = particle_index_l - global_index_begin;
        } else {
            y_update_flag[threadIdx.z][threadIdx.x][j % 2] = 0;
        }


        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < global_index_end)) {
            particle_index_l++;
            y_l = y_vec[particle_index_l];
            f_l[threadIdx.z][threadIdx.x] = input_particles[particle_index_l];
        }

        //parent update loop
        if ((2 * y_p <= j) && ((particle_index_p + 1) < global_index_end_p)) {
            particle_index_p++;
            y_p = y_vec[particle_index_p];
            f_p = input_particles[particle_index_p];
        }


        //update at child level
        if ((y_t <= j) && ((particle_index_t + 1) < global_index_end_t)) {
            particle_index_t++;
            y_t = y_vec_tree[particle_index_t];
            f_t = tree_data[particle_index_t];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[threadIdx.z][threadIdx.x][(j - filter_offset + 2) % 2] == 1) {
            float neighbour_sum = 0;

            LOCALPATCHCONV333(particle_data_output, global_index_begin +
                              y_update_index[threadIdx.z][threadIdx.x][(j + 2 - filter_offset) % 2],
                              threadIdx.z, threadIdx.x, j - 1, neighbour_sum);
        }
        __syncthreads();

    }

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();
    //set the boundary condition (zeros in this case)

    if (y_update_flag[threadIdx.z][threadIdx.x][(y_num - 1) % 2] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;

        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1, neighbour_sum);

    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_min_333(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             const uint16_t* y_vec,
                             const inputType* input_particles,
                             outputType* particle_data_output,
                             const stencilType* stencil,
                             const uint64_t* level_xz_vec_tree,
                             const uint64_t* xz_end_vec_tree,
                             const uint16_t* y_vec_tree,
                             const treeType* tree_data,
                             const int z_num,
                             const int x_num,
                             const int y_num,
                             const int z_num_parent,
                             const int x_num_parent,
                             const int y_num_parent,
                             const int level,
                             const bool* blocks_empty) {

    if(blocks_empty[blockIdx.z * gridDim.x + blockIdx.x]) {
        return;
    }

    const unsigned int N = 4;

    __shared__
    stencilType local_patch[10][10][4]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    //uint16_t y_cache[N] = {0}; // These are local register/private caches
    //uint16_t index_cache[N] = {0}; // These are local register/private caches

    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);


    bool not_ghost = false;

    if ((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)) {
        not_ghost = true;
    }


    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    // current level
    std::size_t xz_start = level_xz_vec[level] + (x_index) + (z_index) * x_num;

    std::size_t global_index_begin = xz_end_vec[xz_start - 1];
    std::size_t global_index_end = xz_end_vec[level];

    std::size_t particle_index_l = global_index_begin;
    std::uint16_t y_l = y_vec[particle_index_l];
    inputType f_l = input_particles[particle_index_l];


    //std::size_t y_block = 1;
    std::uint16_t y_update_flag[2] = {0};
    std::size_t y_update_index[2] = {0};


    xz_start = level_xz_vec_tree[level] + (x_index) + (z_index) * x_num;

    std::size_t global_index_begin_t = xz_end_vec_tree[xz_start - 1];
    std::size_t global_index_end_t = xz_end_vec_tree[xz_start];


    std::size_t particle_index_t = global_index_begin_t;
    std::uint16_t y_t = y_vec_tree[particle_index_t];
    std::float_t f_t = tree_data[particle_index_t];


    if (global_index_begin_t == global_index_end_t) {
        y_t = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin == global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)

    const int filter_offset = 1;

    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        __syncthreads();

        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N] = f_l; //initial update
            y_update_flag[j % 2] = 1;
            y_update_index[j % 2] = particle_index_l;
        } else {
            y_update_flag[j % 2] = 0;
        }

        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < global_index_end)) {
            particle_index_l++;
            y_l = y_vec[particle_index_l];
            f_l = input_particles[particle_index_l];
        }


        //Check if its time to update child level
        if (j == y_t) {
            local_patch[threadIdx.z][threadIdx.x][y_t % N] = f_t; //initial update
        }

        //update at current level
        if ((y_t <= j) && ((particle_index_t + 1) < global_index_end_t)) {
            particle_index_t++;
            y_t = y_vec_tree[particle_index_t];
            f_t = tree_data[particle_index_t];
        }

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[(j - filter_offset + 2) % 2] == 1) {
            float neighbour_sum = 0;

            LOCALPATCHCONV333(particle_data_output, y_update_index[(j + 2 - filter_offset) % 2],
                              threadIdx.z, threadIdx.x, j - 1, neighbour_sum);
        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 1) % 2] == 1) { //the last particle (if it exists)

        float neighbour_sum = 0;

        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1, neighbour_sum);
    }
}


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_555(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             const uint16_t* y_vec,
                             const inputType* input_particles,
                             outputType* output_particles,
                             const stencilType* stencil,
                             const int z_num,
                             const int x_num,
                             const int y_num,
                             const int z_num_parent,
                             const int x_num_parent,
                             const int y_num_parent,
                             const int level,
                             const bool* blocks_empty) {

    if(blocks_empty[blockIdx.z * gridDim.x + blockIdx.x]) {
        return;
    }

    const unsigned int N = 6;

    __shared__
    stencilType local_patch[12][12][6]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);

    bool not_ghost = false;

    if ((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)) {
        not_ghost = true;
    }

    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;
        return; //out of bounds
    }

    int x_index_p = (x_index) / 2;
    int z_index_p = (z_index) / 2;

    // current level
    std::size_t xz_start = level_xz_vec[level] + (x_index) + (z_index) * x_num;
    std::size_t global_index_begin = xz_end_vec[xz_start - 1];
    std::size_t global_index_end = xz_end_vec[xz_start];

    // parent level
    xz_start = level_xz_vec[level - 1] + (x_index_p) + (z_index_p) * x_num_parent;
    std::size_t global_index_begin_p = xz_end_vec[xz_start - 1];
    std::size_t global_index_end_p = xz_end_vec[xz_start];

    //std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = global_index_begin;
    std::uint16_t y_l = y_vec[particle_index_l];
    inputType f_l = input_particles[particle_index_l];


    //parent level variables
    std::size_t particle_index_p = global_index_begin_p;
    std::uint16_t y_p = y_vec[particle_index_p];
    inputType f_p = input_particles[particle_index_p];


    if (global_index_begin_p == global_index_end_p) {
        y_p = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin == global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)
    local_patch[threadIdx.z][threadIdx.x][(N - 2) % N] = 0; //this is at (y-2)

    const int filter_offset = 2;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T
        __syncthreads();
        //Check if its time to update the parent level


        if (j == (2 * y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N] = f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j + 1) % N] = f_p;
        }


        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][j % N] = f_l; //initial update
            y_update_flag[j % 3] = 1;
            y_update_index[j % 3] = particle_index_l;
        } else {
            y_update_flag[j % 3] = 0;
        }

        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < global_index_end)) {
            particle_index_l++;
            y_l = y_vec[particle_index_l];
            f_l = input_particles[particle_index_l];
        }

        //parent update loop
        if ((2 * y_p <= j) && ((particle_index_p + 1) < global_index_end_p)) {
            particle_index_p++;
            y_p = y_vec[particle_index_p];
            f_p = input_particles[particle_index_p];
        }

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[(j - filter_offset + 3) % 3] == 1) {
            float neighbour_sum = 0;
            LOCALPATCHCONV555(output_particles, y_update_index[(j + 3 - filter_offset) % 3],
                              threadIdx.z, threadIdx.x, j - 2, neighbour_sum);
        }
    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 2) % 3] == 1) { //the second to last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV555(output_particles, y_update_index[(y_num + 3 - 2) % 3], threadIdx.z, threadIdx.x,
                          y_num - 2, neighbour_sum);

    }

    __syncthreads();
    local_patch[threadIdx.z][threadIdx.x][(y_num + 1) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 1) % 3] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV555(output_particles, y_update_index[(y_num + 3 - 1) % 3], threadIdx.z, threadIdx.x,
                          y_num - 1, neighbour_sum);
    }


}

template<typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_555(const uint64_t* level_xz_vec,
                                  const uint64_t* xz_end_vec,
                                  const uint16_t* y_vec,
                                  const inputType* input_particles,
                                  outputType* output_particles,
                                  const stencilType* stencil,
                                  const uint64_t* level_xz_vec_tree,
                                  const uint64_t* xz_end_vec_tree,
                                  const uint16_t* y_vec_tree,
                                  const treeType* tree_data,
                                  const int z_num,
                                  const int x_num,
                                  const int y_num,
                                  const int z_num_parent,
                                  const int x_num_parent,
                                  const int y_num_parent,
                                  const int level,
                                  const bool* blocks_empty) {

    if(blocks_empty[blockIdx.z * gridDim.x + blockIdx.x]) {
        return;
    }

    const unsigned int N = 6;

    __shared__
    stencilType local_patch[12][12][6]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);


    bool not_ghost = false;

    if ((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)) {
        not_ghost = true;
    }


    if ((x_index >= x_num) || (x_index < 0)) {
        //set the whole buffer to the boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        //set the whole buffer to the zero boundary condition
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    int x_index_p = (8 * blockIdx.x + threadIdx.x - 2) / 2;
    int z_index_p = (8 * blockIdx.z + threadIdx.z - 2) / 2;

    // current level (apr)
    std::size_t xz_start = level_xz_vec[level] + (x_index) + (z_index) * x_num;
    std::size_t global_index_begin = xz_end_vec[xz_start - 1];
    std::size_t global_index_end = xz_end_vec[xz_start];

    // parent level (apr)
    xz_start = level_xz_vec[level - 1] + (x_index_p) + (z_index_p) * x_num_parent;
    std::size_t global_index_begin_p = xz_end_vec[xz_start - 1];
    std::size_t global_index_end_p = xz_end_vec[xz_start];

    //std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = global_index_begin;
    std::uint16_t y_l = y_vec[particle_index_l];
    inputType f_l = input_particles[particle_index_l];


    //parent level variables
    std::size_t particle_index_p = global_index_begin_p;
    std::uint16_t y_p = y_vec[particle_index_p];
    inputType f_p = input_particles[particle_index_p];

    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    xz_start = level_xz_vec_tree[level] + (x_index) + (z_index) * x_num;

    std::size_t global_index_begin_t = xz_end_vec_tree[xz_start - 1];
    std::size_t global_index_end_t = xz_end_vec_tree[xz_start];

    std::size_t particle_index_t = global_index_begin_t;
    std::uint16_t y_t = y_vec_tree[particle_index_t];
    treeType f_t = tree_data[particle_index_t];

    if (global_index_begin_t == global_index_end_t) {
        y_t = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin_p == global_index_end_p) {
        y_p = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin == global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }

    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)
    local_patch[threadIdx.z][threadIdx.x][(N - 2) % N] = 0; //this is at (y-2)

    const int filter_offset = 2;

    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        //Check if its time to update the parent level
        if (j == (2 * y_p)) {
            local_patch[threadIdx.z][threadIdx.x][(j) % N] = f_p; //initial update
            local_patch[threadIdx.z][threadIdx.x][(j + 1) % N] = f_p;
        }

        //Check if its time to update child level
        if (j == y_t) {
            local_patch[threadIdx.z][threadIdx.x][y_t % N] = f_t; //initial update
        }

        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N] = f_l; //initial update
            y_update_flag[j % 3] = 1;
            y_update_index[j % 3] = particle_index_l;
        } else {
            y_update_flag[j % 3] = 0;
        }


        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < global_index_end)) {
            particle_index_l++;
            y_l = y_vec[particle_index_l];
            f_l = input_particles[particle_index_l];
        }

        //parent update loop
        if ((2 * y_p <= j) && ((particle_index_p + 1) < global_index_end_p)) {
            particle_index_p++;
            y_p = y_vec[particle_index_p];
            f_p = input_particles[particle_index_p];
        }


        //update at child level
        if ((y_t <= j) && ((particle_index_t + 1) < global_index_end_t)) {
            particle_index_t++;
            y_t = y_vec_tree[particle_index_t];
            f_t = tree_data[particle_index_t];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[(j - filter_offset + 3) % 3] == 1) {
            float neighbour_sum = 0;

            LOCALPATCHCONV555(output_particles, y_update_index[(j + 3 - filter_offset) % 3], threadIdx.z,
                              threadIdx.x, j - 2, neighbour_sum);
        }
        __syncthreads();

    }

    //set the boundary condition for last 2 iterations (zeros in this case)
    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 2) % 3] == 1) { //the second to last particle (if it exists)

        float neighbour_sum = 0;

        LOCALPATCHCONV555(output_particles, y_update_index[(y_num + 3 - 2) % 3], threadIdx.z, threadIdx.x,
                          y_num - 2, neighbour_sum);

    }

    __syncthreads();
    local_patch[threadIdx.z][threadIdx.x][(y_num + 1) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 1) % 3] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;

        LOCALPATCHCONV555(output_particles, y_update_index[(y_num + 3 - 1) % 3], threadIdx.z, threadIdx.x,
                          y_num - 1, neighbour_sum);
    }


}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_min_555(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             const uint16_t* y_vec,
                             const inputType* input_particles,
                             outputType* output_particles,
                             const stencilType* stencil,
                             const uint64_t* level_xz_vec_tree,
                             const uint64_t* xz_end_vec_tree,
                             const uint16_t* y_vec_tree,
                             const treeType* tree_data,
                             const int z_num,
                             const int x_num,
                             const int y_num,
                             const int z_num_parent,
                             const int x_num_parent,
                             const int y_num_parent,
                             const int level,
                             const bool* blocks_empty) {

    if(blocks_empty[blockIdx.z * gridDim.x + blockIdx.x]) {
        return;
    }

    const unsigned int N = 6;

    __shared__
    stencilType local_patch[12][12][6]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);


    bool not_ghost = false;

    if ((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)) {
        not_ghost = true;
    }


    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    // current level
    std::size_t xz_start = level_xz_vec[level] + (x_index) + (z_index) * x_num;
    std::size_t global_index_begin = xz_end_vec[xz_start - 1];
    std::size_t global_index_end = xz_end_vec[xz_start];

    //std::size_t y_block = 1;
    std::uint16_t y_update_flag[3] = {0};
    std::size_t y_update_index[3] = {0};

    //current level variables
    std::size_t particle_index_l = global_index_begin;
    std::uint16_t y_l = y_vec[particle_index_l];
    inputType f_l = input_particles[particle_index_l];

    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    xz_start = level_xz_vec_tree[level] + (x_index) + (z_index) * x_num;

    std::size_t global_index_begin_t = xz_end_vec_tree[xz_start - 1];
    std::size_t global_index_end_t = xz_end_vec_tree[xz_start];

    std::size_t particle_index_t = global_index_begin_t;
    std::uint16_t y_t = y_vec_tree[particle_index_t];
    treeType f_t = tree_data[particle_index_t];


    if (global_index_begin_t == global_index_end_t) {
        y_t = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin == global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }


    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)
    local_patch[threadIdx.z][threadIdx.x][(N - 2) % N] = 0; //this is at (y-2)

    const int filter_offset = 2;


    for (int j = 0; j < (y_num); ++j) {

        //Update steps for P->T

        /*
         *
         * Current Level Update
         *
         */

        __syncthreads();

        //Check if its time to update current level
        if (j == y_l) {
            local_patch[threadIdx.z][threadIdx.x][y_l % N] = f_l; //initial update
            y_update_flag[j % 3] = 1;
            y_update_index[j % 3] = particle_index_l;
        } else {
            y_update_flag[j % 3] = 0;
        }

        //update at current level
        if ((y_l <= j) && ((particle_index_l + 1) < global_index_end)) {
            particle_index_l++;
            y_l = y_vec[particle_index_l];
            f_l = input_particles[particle_index_l];
        }

        /*
         *
         * Child Level Update
         *
         */


        //Check if its time to update current level
        if (j == y_t) {
            local_patch[threadIdx.z][threadIdx.x][y_t % N] = f_t; //initial update
        }

        //update at current level
        if ((y_t <= j) && ((particle_index_t + 1) < global_index_end_t)) {
            particle_index_t++;
            y_t = y_vec_tree[particle_index_t];
            f_t = tree_data[particle_index_t];
        }


        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        if (y_update_flag[(j - filter_offset + 3) % 3] == 1) {
            float neighbour_sum = 0;

            LOCALPATCHCONV555(output_particles, y_update_index[(j + 3 - filter_offset) % 3], threadIdx.z,
                              threadIdx.x, j - 2, neighbour_sum);
        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num + 3 - 2) % 3] == 1) { //the second to last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV555(output_particles, y_update_index[(y_num + 3 - 2) % 3], threadIdx.z, threadIdx.x,
                          y_num - 2, neighbour_sum);
    }

    __syncthreads();
    local_patch[threadIdx.z][threadIdx.x][(y_num + 1) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num + 3 - 1) % 3] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV555(output_particles, y_update_index[(y_num + 3 - 1) % 3], threadIdx.z, threadIdx.x,
                          y_num - 1, neighbour_sum);
    }
}



template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333(const inputType* input_image,
                               outputType* output_image,
                               const stencilType* stencil,
                               const int z_num,
                               const int x_num,
                               const int y_num){


    // This is block wise shared memory this is assuming an 8*8 block with pad()

    bool not_ghost = false;

    if ((threadIdx.x > 0) && (threadIdx.x < 9) && (threadIdx.z > 0) && (threadIdx.z < 9)) {
        not_ghost = true;
    }

    int x_index = (8 * blockIdx.x + threadIdx.x - 1);
    int z_index = (8 * blockIdx.z + threadIdx.z - 1);

    const unsigned int N = 4;

    __shared__
    stencilType local_patch[10][10][N];

    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        return; //out of bounds
    }

    std::size_t row_begin = z_index * x_num * y_num + x_index * y_num;

    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)
    local_patch[threadIdx.z][threadIdx.x][0] = input_image[row_begin]; //initial update

    for (size_t j = 1; j < y_num; ++j) {

        //Update steps for P->T
        __syncthreads();

        local_patch[threadIdx.z][threadIdx.x][j % N] = input_image[row_begin + j]; //initial update

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        float neighbour_sum = 0;
        LOCALPATCHCONV333(output_image, (row_begin + j - 1), threadIdx.z, threadIdx.x, j - 1, neighbour_sum)
    }

    __syncthreads();

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    float neighbour_sum = 0;
    LOCALPATCHCONV333(output_image, row_begin + y_num - 1, threadIdx.z, threadIdx.x, y_num - 1, neighbour_sum)
}


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_555(const inputType* input_image,
                               outputType* output_image,
                               const stencilType* stencil,
                               const int z_num,
                               const int x_num,
                               const int y_num){


    // This is block wise shared memory this is assuming an 8*8 block with pad()

    bool not_ghost = false;

    if ((threadIdx.x > 1) && (threadIdx.x < 10) && (threadIdx.z > 1) && (threadIdx.z < 10)) {
        not_ghost = true;
    }

    int x_index = (8 * blockIdx.x + threadIdx.x - 2);
    int z_index = (8 * blockIdx.z + threadIdx.z - 2);

    const unsigned int N = 6;

    __shared__
    stencilType local_patch[12][12][N];

    if ((x_index >= x_num) || (x_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    if ((z_index >= z_num) || (z_index < 0)) {
        local_patch[threadIdx.z][threadIdx.x][0] = 0; //this is at (y-1)
        local_patch[threadIdx.z][threadIdx.x][1] = 0;
        local_patch[threadIdx.z][threadIdx.x][2] = 0;
        local_patch[threadIdx.z][threadIdx.x][3] = 0;
        local_patch[threadIdx.z][threadIdx.x][4] = 0;
        local_patch[threadIdx.z][threadIdx.x][5] = 0;

        return; //out of bounds
    }

    std::size_t row_begin = z_index * x_num * y_num + x_index * y_num;

    //BOUNDARY CONDITIONS
    local_patch[threadIdx.z][threadIdx.x][(N - 1) % N] = 0; //this is at (y-1)
    local_patch[threadIdx.z][threadIdx.x][(N - 2) % N] = 0;

    //initial update
    local_patch[threadIdx.z][threadIdx.x][0 % N] = input_image[row_begin];
    local_patch[threadIdx.z][threadIdx.x][1 % N] = input_image[row_begin+1];


    for (size_t j = 2; j < y_num; ++j) {

        //Update steps for P->T
        __syncthreads();

        local_patch[threadIdx.z][threadIdx.x][j % N] = input_image[row_begin + j]; //fill in next value

        __syncthreads();
        //COMPUTE THE T->P from shared memory, this is lagged by the size of the filter

        float neighbour_sum = 0;
        LOCALPATCHCONV555(output_image, (row_begin + j - 2), threadIdx.z, threadIdx.x, j - 2, neighbour_sum)
    }

    __syncthreads();

    // now do the last two iterations

    //set the boundary condition (zeros in this case)
    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    float neighbour_sum = 0;
    LOCALPATCHCONV555(output_image, row_begin + y_num - 2, threadIdx.z, threadIdx.x, y_num - 2, neighbour_sum)

    //set the boundary condition (zeros in this case)
    local_patch[threadIdx.z][threadIdx.x][(y_num + 1) % N] = 0;
    __syncthreads();

    neighbour_sum = 0;
    LOCALPATCHCONV555(output_image, row_begin + y_num - 1, threadIdx.z, threadIdx.x, y_num - 1, neighbour_sum)
}