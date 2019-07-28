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
timings convolve_pixel_333(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil) {

    assert(stencil.mesh.size() == 27);

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
timings convolve_pixel_555(PixelData<inputType>& input, PixelData<outputType>& output, PixelData<stencilType>& stencil) {

    assert(stencil.mesh.size() == 125);

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

void compute_ne_rows(GPUAccessHelper& tree_access,std::vector<int>& ne_counter,std::vector<int>& ne_rows) {
    ne_counter.resize(tree_access.level_max() + 3);

    int z = 0;
    int x = 0;

    for (int level = (tree_access.level_min() + 1); level <= (tree_access.level_max() + 1); ++level) {

        auto level_start = tree_access.linearAccess->level_xz_vec[level - 1];

        ne_counter[level] = ne_rows.size();

        for (z = 0; z < tree_access.z_num(level - 1); z++) {
            for (x = 0; x < tree_access.x_num(level - 1); ++x) {

                auto offset = x + z * tree_access.x_num(level - 1);
                auto xz_start = level_start + offset;

//intialize
                auto begin_index = tree_access.linearAccess->xz_end_vec[xz_start - 1];
                auto end_index = tree_access.linearAccess->xz_end_vec[xz_start];

                if (begin_index < end_index) {
                    ne_rows.push_back(x + z * tree_access.x_num(level - 1));
                }
            }
        }
    }
    ne_counter.back() = ne_rows.size();
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
                                    std::vector<outputType>& output, std::vector<stencilType>& stencil, std::vector<treeType>& tree_data){
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel
     *
     *  conv_stencil needs to have 27 entries
     */

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 27);

    const int blockSize = 8;

    timings ret;
    ret.lvl_timings.resize(access.level_max() - access.level_min() + 1);

    APRTimer timer(false);
    APRTimer timer2(false);

    timer.start_timer("init arrays");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();

    std::vector<int> ne_rows; //non empty rows
    std::vector<int> ne_counter; //non empty rows

    compute_ne_rows(tree_access,ne_counter,ne_rows);
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu(ne_rows.data(), ne_rows.size());

    timer.start_timer("transfer H2D");

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    size_t max_num_blocks = ((access.x_num(access.level_max()) + blockSize - 1) / blockSize) * ((access.z_num(access.level_max()) + blockSize - 1) / blockSize);
    ScopedCudaMemHandler<bool*, JUST_ALLOC> blocks_empty(NULL, max_num_blocks);


    ne_rows_gpu.copyH2D();

    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    cudaDeviceSynchronize();
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("fill tree");
    downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get(),ne_rows_gpu.get(),ne_counter);
    cudaDeviceSynchronize();
    timer.stop_timer();

    ret.fill_tree = timer.timings.back();

    timer.start_timer("run kernels");

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        timer2.start_timer("convolve_dlvl_" + std::to_string(access.level_max() - level));
        int x_blocks = (access.x_num(level) + blockSize - 1) / blockSize;
        int z_blocks = (access.z_num(level) + blockSize - 1) / blockSize;

        dim3 blocks_l(x_blocks, 1, z_blocks);
        dim3 threads_l(blockSize, 1, blockSize);

        check_blocks<blockSize><<<blocks_l, threads_l>>>(access.get_level_xz_vec_ptr(),
                                            access.get_xz_end_vec_ptr(),
                                            blocks_empty.get(),
                                            level,
                                            access.x_num(level));

        threads_l = {blockSize+2, 1, blockSize+2};

        if (level == access.level_min()) {
            conv_min_333<blockSize> << < blocks_l, threads_l >> >( access.get_level_xz_vec_ptr(),
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
            conv_max_333<blockSize> << < blocks_l, threads_l >> >( access.get_level_xz_vec_ptr(),
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

//            dim3 blck(access.x_num(level), 1, access.y_num(level));
//            dim3 thd(6, 3, 3);
//
//            conv_max_333_alt<<<blck, thd>>>(access.get_level_xz_vec_ptr(),
//                                            access.get_xz_end_vec_ptr(),
//                                            access.get_y_vec_ptr(),
//                                            input_gpu.get(),
//                                            output_gpu.get(),
//                                            stencil_gpu.get(),
//                                            access.z_num(level),
//                                            access.x_num(level),
//                                            access.y_num(level),
//                                            tree_access.z_num(level-1),
//                                            tree_access.x_num(level-1),
//                                            level);

        } else {
            conv_interior_333<blockSize> << < blocks_l, threads_l >> >(access.get_level_xz_vec_ptr(),
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
        timer2.stop_timer();
        ret.lvl_timings[access.level_max() - level] = timer2.timings.back();
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


template<typename inputType, typename outputType, typename stencilType>
void run_max_333_new(GPUAccessHelper &access, inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu) {

    int level = access.level_max();

    dim3 blocks_l(access.x_num(level), 1, access.y_num(level));
    dim3 threads_l(6, 3, 3);

    conv_max_333_alt<<<blocks_l, threads_l>>>(access.get_level_xz_vec_ptr(),
                                            access.get_xz_end_vec_ptr(),
                                            access.get_y_vec_ptr(),
                                            input_gpu,
                                            output_gpu,
                                            stencil_gpu,
                                            access.z_num(level),
                                            access.x_num(level),
                                            access.y_num(level),
                                            access.z_num(level-1),
                                            access.x_num(level-1),
                                            level);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}

template<typename inputType, typename outputType, typename stencilType>
void run_max_333_old(GPUAccessHelper& access, inputType* input_gpu, outputType* output_gpu, stencilType* stencil_gpu, bool* blocks_empty) {

    int level = access.level_max();

    const int blockSize = 8;

    int x_blocks = (access.x_num(level) + blockSize - 1) / blockSize;
    int z_blocks = (access.z_num(level) + blockSize - 1) / blockSize;

    dim3 blocks_l(x_blocks, 1, z_blocks);
    dim3 threads_l(blockSize, 1, blockSize);

    check_blocks<blockSize><<<blocks_l, threads_l>>>(access.get_level_xz_vec_ptr(),
                                                    access.get_xz_end_vec_ptr(),
                                                    blocks_empty,
                                                    level,
                                                    access.x_num(level));

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    threads_l = {blockSize+2, 1, blockSize+2};

    conv_max_333<blockSize> << < blocks_l, threads_l >> >(  access.get_level_xz_vec_ptr(),
                                                            access.get_xz_end_vec_ptr(),
                                                            access.get_y_vec_ptr(),
                                                            input_gpu,
                                                            output_gpu,
                                                            stencil_gpu,
                                                            access.z_num(level),
                                                            access.x_num(level),
                                                            access.y_num(level),
                                                            access.z_num(level-1),
                                                            access.x_num(level-1),
                                                            access.y_num(level-1),
                                                            level,
                                                            blocks_empty );

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, std::vector<inputType>& input,
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


    std::vector<int> ne_rows; //non empty rows
    std::vector<int> ne_counter; //non empty rows

    compute_ne_rows(tree_access,ne_counter,ne_rows);
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu(ne_rows.data(), ne_rows.size());

    timer.start_timer("transfer H2D");

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    size_t max_num_blocks = ((access.x_num(access.level_max()) + 8 - 1) / 8) * ((access.z_num(access.level_max()) + 8 - 1) / 8);
    ScopedCudaMemHandler<bool*, JUST_ALLOC> blocks_empty(NULL, max_num_blocks);

    /// copy input and stencil to the GPU
    ne_rows_gpu.copyH2D();

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    cudaDeviceSynchronize();

    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("fill tree");
    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(),ne_rows_gpu.get(),ne_counter);
    cudaDeviceSynchronize();
    timer.stop_timer();

    ret.fill_tree = timer.timings.back();

    timer.start_timer("run kernels");

    for (int level = access.level_max(); level >= access.level_min(); --level) {

        int x_blocks = (access.x_num(level) + 8 - 1) / 8;
        int z_blocks = (access.z_num(level) + 8 - 1) / 8;

        dim3 blocks_l(x_blocks, 1, z_blocks);
        dim3 thd_l(8, 1, 8);

        check_blocks<8><<<blocks_l, thd_l>>>(access.get_level_xz_vec_ptr(),
                                        access.get_xz_end_vec_ptr(),
                                        blocks_empty.get(),
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


void run_check_blocks(GPUAccessHelper& access, bool* blocks_empty) {

    for(int level = access.level_max(); level >= access.level_min(); --level) {

        int x_blocks = (access.x_num(level) + 8 - 1) / 8;
        int z_blocks = (access.z_num(level) + 8 - 1) / 8;

        dim3 blocks_l(x_blocks, 1, z_blocks);
        dim3 threads_l(8, 1, 8);

        check_blocks<8><<< blocks_l, threads_l >>>(access.get_level_xz_vec_ptr(),
                                                    access.get_xz_end_vec_ptr(),
                                                    blocks_empty,
                                                    level,
                                                    access.x_num(level));
    }
}


template<unsigned int blockSize>
__device__ void warpReduce(volatile bool* sdata, int tid) {
    if(blockSize >= 64) {if(tid+32 < blockSize) {sdata[tid] *= sdata[tid+32];}}
    if(blockSize >= 32) {if(tid+16 < blockSize){sdata[tid] *= sdata[tid+16];}}
    if(blockSize >= 16) {if(tid+8 < blockSize){sdata[tid] *= sdata[tid+8];}}
    if(blockSize >= 8) {if(tid+4 < blockSize){sdata[tid] *= sdata[tid+4];}}
    if(blockSize >= 4) {if(tid+2 < blockSize){sdata[tid] *= sdata[tid+2];}}
    if(blockSize >= 2) {if(tid+1 < blockSize){sdata[tid] *= sdata[tid+1];}}
}

template<unsigned int blockSize>
__global__ void check_blocks(const uint64_t* level_xz_vec,
                             const uint64_t* xz_end_vec,
                             bool* blocks_empty,
                             const int level,
                             const int x_num) {

    const int tid = threadIdx.z * blockDim.x + threadIdx.x;

    __shared__
    bool shared_block[blockSize * blockSize];

    int x_start = blockSize * blockIdx.x;
    int z_start = blockSize * blockIdx.z;

    size_t xz_start = (x_start + threadIdx.x) + (z_start + threadIdx.z) * x_num + level_xz_vec[level];
    size_t global_index_begin = xz_end_vec[xz_start-1];
    size_t global_index_end = xz_end_vec[xz_start];

    shared_block[tid] = (global_index_begin >= global_index_end);
    __syncthreads();

    //if(tid < 32) warpReduce<blockSize*blockSize>(shared_block, tid);
    int i=ceil((float)blockSize*blockSize/2);
    while(i != 0) {
        if(tid + i < blockSize*blockSize && tid < i){
            shared_block[tid] *= shared_block[tid+i];
        }
        i /= 2;
        __syncthreads();
    }

    if(tid == 0){
        blocks_empty[blockIdx.z * gridDim.x + blockIdx.x] = shared_block[0];
    }
}


template<unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
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

    if ((threadIdx.x > 0) && (threadIdx.x < (blockDim.x-1)) && (threadIdx.z > 0) && (threadIdx.z < (blockDim.z-1))) {
        not_ghost = true;
    }

    int x_index = (blockSize * blockIdx.x + threadIdx.x - 1);
    int z_index = (blockSize * blockIdx.z + threadIdx.z - 1);

    const unsigned int N = 4;

    __shared__
    stencilType local_patch[blockSize+2][blockSize+2][N];

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
    std::uint16_t y_update_flag[blockSize+2][blockSize+2][2];
    __shared__
    std::uint16_t y_update_index[blockSize+2][blockSize+2][2];

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
                              threadIdx.z, threadIdx.x, j - 1, neighbour_sum)

        }
    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[threadIdx.z][threadIdx.x][(y_num - 1) % 2] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;
        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1,
                          neighbour_sum)

    }
}



template<unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void __launch_bounds__(100, 16)
conv_interior_333(const uint64_t* level_xz_vec,
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

    const int x_index = (blockSize * blockIdx.x + threadIdx.x - 1);
    const int z_index = (blockSize * blockIdx.z + threadIdx.z - 1);


    const bool not_ghost = (threadIdx.x > 0) && (threadIdx.x < (blockDim.x-1)) && (threadIdx.z > 0) && (threadIdx.z < (blockDim.z-1));

    __shared__
    stencilType local_patch[blockSize+2][blockSize+2][N]; // This is block wise shared memory this is assuming an 8*8 block with pad()

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

    const int x_index_p = (blockSize * blockIdx.x + threadIdx.x - 1) / 2;
    const int z_index_p = (blockSize * blockIdx.z + threadIdx.z - 1) / 2;

    // current level
    std::size_t xz_start = level_xz_vec[level] + (x_index) + (z_index) * x_num;
    const std::size_t global_index_begin = xz_end_vec[xz_start - 1];
    const std::size_t global_index_end = xz_end_vec[xz_start];

    std::size_t particle_index_l = global_index_begin;
    std::uint16_t y_l = y_vec[particle_index_l];
    //inputType f_l = input_particles[particle_index_l];

    // parent level, level - 1, one resolution lower (coarser)
    xz_start = level_xz_vec[level - 1] + (x_index_p) + (z_index_p) * x_num_parent;
    const size_t global_index_begin_p = xz_end_vec[xz_start - 1];
    const size_t global_index_end_p = xz_end_vec[xz_start];

    //parent level variables
    std::size_t particle_index_p = global_index_begin_p;
    std::uint16_t y_p = y_vec[particle_index_p];
    inputType f_p = input_particles[particle_index_p];


    /*
    * Child level variable initialization, using 'Tree'
    * This is the same row as the current level
    */

    xz_start = level_xz_vec_tree[level] + (x_index) + (z_index) * x_num;

    const std::size_t global_index_begin_t = xz_end_vec_tree[xz_start - 1];
    const std::size_t global_index_end_t = xz_end_vec_tree[xz_start];

    std::size_t particle_index_t = global_index_begin_t;
    std::uint16_t y_t = y_vec_tree[particle_index_t];
    treeType f_t = tree_data[particle_index_t];


    if (global_index_begin_t == global_index_end_t) {
        y_t = y_num + 1;//no particles don't do anything
    }

    if (global_index_begin_p == global_index_end_p) {
        y_p = y_num_parent + 1;//no particles don't do anything
    }

    if (global_index_begin == global_index_end) {
        y_l = y_num + 1;//no particles don't do anything
    }

    __shared__
    std::uint16_t y_update_flag[blockSize+2][blockSize+2][2];
    __shared__
    std::uint16_t y_update_index[blockSize+2][blockSize+2][2];

    __shared__
    std::uint16_t f_l[blockSize+2][blockSize+2];

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
                              threadIdx.z, threadIdx.x, j - 1, neighbour_sum)
        }
        __syncthreads();

    }

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();
    //set the boundary condition (zeros in this case)

    if (y_update_flag[threadIdx.z][threadIdx.x][(y_num - 1) % 2] == 1) { //the last particle (if it exists)
        float neighbour_sum = 0;

        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1, neighbour_sum)

    }
}


template<unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
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
    stencilType local_patch[blockSize+2][blockSize+2][N]; // This is block wise shared memory this is assuming an 8*8 block with pad()

    //uint16_t y_cache[N] = {0}; // These are local register/private caches
    //uint16_t index_cache[N] = {0}; // These are local register/private caches

    int x_index = (blockSize * blockIdx.x + threadIdx.x - 1);
    int z_index = (blockSize * blockIdx.z + threadIdx.z - 1);


    bool not_ghost = false;

    if ((threadIdx.x > 0) && (threadIdx.x < (blockDim.x-1)) && (threadIdx.z > 0) && (threadIdx.z < (blockDim.z-1))) {
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
                              threadIdx.z, threadIdx.x, j - 1, neighbour_sum)
        }

    }

    //set the boundary condition (zeros in this case)

    local_patch[threadIdx.z][threadIdx.x][(y_num) % N] = 0;
    __syncthreads();

    if (y_update_flag[(y_num - 1) % 2] == 1) { //the last particle (if it exists)

        float neighbour_sum = 0;

        LOCALPATCHCONV333(particle_data_output, particle_index_l, threadIdx.z, threadIdx.x, y_num - 1, neighbour_sum)
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


template<typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_alt(const uint64_t* level_xz_vec,
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
                                 const int level) {

    // call with blockDim (6, 3, 3)

    const int local_x = threadIdx.x % 3;

    int x_index = blockIdx.x - 1 + local_x;
    int z_index = blockIdx.z - 1 + threadIdx.z;
    const bool parent = threadIdx.x > 2;

    __shared__ stencilType local_patch[27]; //make it 1D and parallel-reduce?
    __shared__ size_t loop_indices[2];

    const int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
    if(tid < 27) {
        local_patch[tid] = 0;
    }

    if( (x_index < 0) || (z_index < 0) ) {
        return;
    }

    if(parent && ( ((x_index/2) >= x_num_parent) || ((z_index/2) >= z_num_parent) )) {
        return;
    } else if( !parent && ( (x_index >= x_num) || (z_index >= z_num) )) {
        return;
    }

    size_t xz_start, global_index_begin, global_index_end;


    if(parent) {
        x_index /= 2;
        z_index /=2;

        xz_start = x_index + z_index * x_num_parent + level_xz_vec[level - 1];
        global_index_begin = xz_end_vec[xz_start-1];
        global_index_end = xz_end_vec[xz_start];

    } else {
        xz_start = x_index + z_index * x_num + level_xz_vec[level];
        global_index_begin = xz_end_vec[xz_start-1];
        global_index_end = xz_end_vec[xz_start];
    }

    if( (threadIdx.x==1) && (threadIdx.y==1) && (threadIdx.z==1) ) {
        loop_indices[0] = global_index_begin;
        loop_indices[1] = global_index_end;
    }
    __syncthreads();

    size_t current_index = global_index_begin;
    int y = y_vec[current_index];
    current_index++;

    int factor = parent ? 2 : 1;


    for(size_t index = loop_indices[0]; index < loop_indices[1]; ++index) {

        int target_y = ((int)(y_vec[index] + threadIdx.y) - 1) / factor;

//        if(!parent && ( (target_y < 0) || (target_y >= y_num) )) {
//            local_patch[threadIdx.z * 9 + local_x * 3 + threadIdx.y] = 0;
//        }

        while( (y < target_y) && (current_index < global_index_end) ) {
            y = y_vec[current_index];
            current_index++;
        }

        if( (y == target_y) && (current_index < global_index_end)) {
            local_patch[threadIdx.z * 9 + local_x * 3 + threadIdx.y] = input_particles[current_index];
        }

        __syncthreads();

        if( (threadIdx.x == 1) && (threadIdx.y == 1) && (threadIdx.z == 1)) {
            stencilType neigh_sum = 0;
            for(auto j : local_patch) { //(int j = 0; j < 27; ++j) {
                neigh_sum += j;//local_patch[j];
            }
            particle_data_output[index] = neigh_sum;
        }
        __syncthreads();

//        /// reduce sum to collect result
//        int i=14; // ceil(27 / 2)
//        while(i != 0) {
//            if( (tid + i < 27) && (tid < i) ){
//                local_patch[tid] += local_patch[tid+i];
//            }
//            i /= 2;
//            __syncthreads();
//        }
//
//        if(tid==0) {
//            particle_data_output[index] = local_patch[0];
//        }
    } // for index
}


/// force template instantiation for some different type combinations
//pixels 333
template timings convolve_pixel_333(PixelData<uint16_t>&, PixelData<float>&, PixelData<float>&);
template timings convolve_pixel_333(PixelData<uint16_t>&, PixelData<double>&, PixelData<double>&);
template timings convolve_pixel_333(PixelData<float>&, PixelData<float>&, PixelData<float>&);
//pixels 555
template timings convolve_pixel_555(PixelData<uint16_t>&, PixelData<float>&, PixelData<float>&);
template timings convolve_pixel_555(PixelData<uint16_t>&, PixelData<double>&, PixelData<double>&);
template timings convolve_pixel_555(PixelData<float>&, PixelData<float>&, PixelData<float>&);
//apr 333
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<float>&, std::vector<float>&);
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<double>&, std::vector<float>&);
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<double>&, std::vector<double>&);
//apr 555
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<float>&, std::vector<float>&);
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<double>&, std::vector<float>&);
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, std::vector<uint16_t>&, std::vector<double>&, std::vector<double>&, std::vector<double>&);

/// play
template void run_max_333_new(GPUAccessHelper&, uint16_t*, float*, float*);
template void run_max_333_old(GPUAccessHelper&, uint16_t*, float*, float*, bool*);