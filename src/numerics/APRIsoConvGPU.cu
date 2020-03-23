//
// Created by cheesema on 09.04.18.
//
#include "io/TiffUtils.hpp"

#include "APRIsoConvGPU.hpp"
#include <cuda_runtime_api.h>

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

#ifdef __CUDACC__
    #define L(x,y) __launch_bounds__(x,y)
#else
    #define L(x,y)
#endif


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
    if(output.size() < input.size()) {
        output.init(input);
    }
    timer.stop_timer();

    timer.start_timer("allocation");
    /// allocate GPU memory
    ScopedCudaMemHandler<PixelData<inputType>, JUST_ALLOC> input_gpu(input);
    ScopedCudaMemHandler<PixelData<outputType>, JUST_ALLOC> output_gpu(output);
    ScopedCudaMemHandler<PixelData<stencilType>, JUST_ALLOC> stencil_gpu(stencil);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();

    ret.allocation = timer.timings.back();

    timer.start_timer("transfer H2D");
    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    timer.start_timer("run kernel");

    const int chunkSize = 32;
    const int blockSize = 4;

    int x_blocks = (input.x_num + 1) / 2;
    int z_blocks = (input.z_num + 1) / 2;

    dim3 blocks_l(x_blocks, 1, z_blocks);
    dim3 threads_l(chunkSize, blockSize, blockSize);

    conv_pixel_333_chunked<chunkSize, blockSize><<< blocks_l, threads_l >>>(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.z_num, input.x_num, input.y_num);

    error_check( cudaDeviceSynchronize() )
    error_check( cudaPeekAtLastError() )

    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
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
    if(output.size() < input.size()) {
        output.init(input);
    }
    timer.stop_timer();

    timer.start_timer("allocation");
    /// allocate GPU memory
    ScopedCudaMemHandler<PixelData<inputType>, JUST_ALLOC> input_gpu(input);
    ScopedCudaMemHandler<PixelData<outputType>, JUST_ALLOC> output_gpu(output);
    ScopedCudaMemHandler<PixelData<stencilType>, JUST_ALLOC> stencil_gpu(stencil);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();

    ret.allocation = timer.timings.back();

    timer.start_timer("transfer H2D");
    /// copy input and stencil to the GPU
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();

    ret.transfer_H2D = timer.timings.back();

    timer.start_timer("run kernel");

    const int chunkSize = 16;
    const int blockSize = 8;

    int x_blocks = (input.x_num + blockSize - 5) / (blockSize-4);
    int z_blocks = (input.z_num + blockSize - 5) / (blockSize-4);

    dim3 blocks_l(x_blocks, 1, z_blocks);
    dim3 threads_l(chunkSize, blockSize, blockSize);


    conv_pixel_555_chunked<chunkSize, blockSize><<< blocks_l, threads_l >>>(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.z_num, input.x_num, input.y_num);

    error_check( cudaDeviceSynchronize() )
    //error_check( cudaPeekAtLastError() )

    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


void compute_ne_rows_interior(GPUAccessHelper& access,VectorData<int>& ne_counter,VectorData<int>& ne_rows) {
    ne_counter.resize(access.level_max()+1);

    int z = 0;
    int x = 0;

    uint64_t counter = 0;

    for (int level = (access.level_min()); level <= (access.level_max() - 1); ++level) {

        auto level_start = access.linearAccess->level_xz_vec[level];

        ne_counter[level] = counter;

        for (z = 0; z < access.z_num(level-1); z++) {
            for (x = 0; x < access.x_num(level-1); ++x) {

                bool nonempty = false;

                for( int ix = 0; ix <= 1; ++ix ){
                    for( int iz = 0; iz <= 1; ++iz ) {
                        auto offset = 2*x + ix + (2*z + iz) * access.x_num(level);
                        auto xz_start = level_start + offset;

                        auto begin_index = access.linearAccess->xz_end_vec[xz_start - 1];
                        auto end_index = access.linearAccess->xz_end_vec[xz_start];

                        if(begin_index < end_index) {
                            nonempty = true;
                        }
                    }
                }

                if (nonempty) {
                    counter++;
                }
            }
        }
    }

    ne_rows.resize(counter);
    counter = 0;

    for (int level = (access.level_min()); level <= (access.level_max() - 1); ++level) {

        auto level_start = access.linearAccess->level_xz_vec[level];

        for (z = 0; z < access.z_num(level-1); z++) {
            for (x = 0; x < access.x_num(level-1); ++x) {

                bool nonempty = false;

                for( int ix = 0; ix <= 1; ++ix ){
                    for( int iz = 0; iz <= 1; ++iz ) {
                        auto offset = 2*x + ix + (2*z + iz) * access.x_num(level);
                        auto xz_start = level_start + offset;

                        auto begin_index = access.linearAccess->xz_end_vec[xz_start - 1];
                        auto end_index = access.linearAccess->xz_end_vec[xz_start];

                        if(begin_index < end_index) {
                            nonempty = true;
                        }
                    }
                }

                if (nonempty) {
                    ne_rows[counter] = (x + z * access.x_num(level-1));
                    counter++;
                }
            }
        }
    }

    ne_counter.back() = ne_rows.size();
}


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


void compute_ne_rows_new(GPUAccessHelper& access, VectorData<int>& ne_counter, VectorData<int>& ne_rows, int block_size = 2) {
    ne_counter.resize(access.level_max()+2);

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


/**
 * Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel. Initializes the necessary memory on
 * the GPU and copies the result back to the host after computation.
 *
 * @param access
 * @param tree_access
 * @param input
 * @param output
 * @param stencil
 * @param tree_data
 * @return
 */
template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                               VectorData<stencilType>& stencil, VectorData<treeType>& tree_data){
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel
     *  conv_stencil needs to have 27 entries, with element (x, y, z) corresponding to index z*9 + x*3 + y
     */

    APRTimer timer(false);
    timings ret;

    timer.start_timer("initialize GPU access (apr and tree)");
    tree_access.init_gpu();
    access.init_gpu(tree_access);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.init_access = timer.timings.back();

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 27);

    timer.start_timer("host data resize");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();

    timer.start_timer("compute nonempty rows");
    VectorData<int> ne_rows_ds; //non empty rows
    VectorData<int> ne_counter_ds; //non empty rows
    compute_ne_rows(tree_access, ne_counter_ds, ne_rows_ds);

    VectorData<int> ne_rows;
    VectorData<int> ne_counter;
    compute_ne_rows_new(access, ne_counter, ne_rows, 2);
    timer.stop_timer();
    ret.compute_ne_rows = timer.timings.back();

    /// allocate GPU memory
    timer.start_timer("allocate GPU memory");
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu(ne_rows_ds.data(), ne_rows_ds.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu(ne_rows.data(), ne_rows.size());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.allocation = timer.timings.back();

    timer.start_timer("transfer H2D");
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();
    ne_rows_ds_gpu.copyH2D();
    ne_rows_gpu.copyH2D();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("downsample (fill tree)");
    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.fill_tree = timer.timings.back();

    timer.start_timer("run_kernels");
    isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter, bool downsample_stencil, stencilType pad_value){

    const int blockSize = 4;
    const int chunkSize = 32;
    size_t stencil_offset = 0;

    for (int level = access.level_max(); level > access.level_min(); --level) {

        size_t ne_sz = ne_counter[level+1] - ne_counter[level];
        size_t offset = ne_counter[level];

        if( ne_sz == 0) {
            if(downsample_stencil) {
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
                                           ne_rows_gpu + offset,
                                           pad_value);

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
                                          ne_rows_gpu + offset,
                                          pad_value);
        }

        if(downsample_stencil) {
            stencil_offset += 27;
        }
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                               VectorData<outputType>& output, VectorData<stencilType>& stencil, VectorData<treeType>& tree_data){
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel
     *  conv_stencil needs to have 27 entries, with element (x, y, z) corresponding to index z*9 + x*3 + y
     */

    APRTimer timer(false);
    timings ret;

    timer.start_timer("initialize GPU access (apr and tree)");
    tree_access.init_gpu();
    access.init_gpu(tree_access);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.init_access = timer.timings.back();

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 27);

    timer.start_timer("host data resize");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();

    /// allocate GPU memory
    timer.start_timer("allocate GPU memory");
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.allocation = timer.timings.back();

    timer.start_timer("transfer H2D");
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("downsample");
    downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.fill_tree = timer.timings.back();

    timer.start_timer("run_kernels");
    isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    error_check( cudaDeviceSynchronize() )
    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_333(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                            outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu, bool downsample_stencil, stencilType pad_value){

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
                                           level,
                                           pad_value);

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
                                          pad_value);
        }
        if(downsample_stencil) {
            stencil_offset += 27;
        }
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_333_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input, VectorData<outputType>& output,
                                  PixelData<stencilType>& stencil, VectorData<treeType>& tree_data, const bool use_ne_rows, const bool normalize_stencil){
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 3x3x3 kernel
     *  conv_stencil needs to have 27 entries, with element (x, y, z) corresponding to index z*9 + x*3 + y
     */
    assert(input.size() == access.total_number_particles());
    assert(stencil.mesh.size() == 27);

    APRTimer timer(false);
    timings ret;

    timer.start_timer("initialize GPU access (apr and tree)");
    tree_access.init_gpu();
    access.init_gpu(tree_access);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.init_access = timer.timings.back();

    timer.start_timer("host data resize");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();

    timer.start_timer("downsample stencil");
    VectorData<stencilType> stencil_vec;
    get_downsampled_stencils(stencil, stencil_vec, access.level_max() - access.level_min(), normalize_stencil);
    timer.stop_timer();
    ret.downsample_stencil = timer.timings.back();

    timer.start_timer("compute nonempty rows");
    VectorData<int> ne_rows_ds; //non empty rows
    VectorData<int> ne_counter_ds; //non empty rows
    VectorData<int> ne_rows;
    VectorData<int> ne_counter;

    if(use_ne_rows) {
        compute_ne_rows(tree_access, ne_counter_ds, ne_rows_ds);
        compute_ne_rows_new(access, ne_counter, ne_rows, 2);
    }
    timer.stop_timer();
    ret.compute_ne_rows = timer.timings.back();

    /// allocate GPU memory
    timer.start_timer("allocate GPU memory");
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil_vec.data(), stencil.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu(ne_rows_ds.data(), ne_rows_ds.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu(ne_rows.data(), ne_rows.size());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.allocation = timer.timings.back();

    timer.start_timer("transfer H2D");
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();
    ne_rows_ds_gpu.copyH2D();
    ne_rows_gpu.copyH2D();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("downsample (fill tree)");
    if(use_ne_rows) {
        downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    } else {
        downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    }
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.fill_tree = timer.timings.back();

    timer.start_timer("run_kernels");
    if(use_ne_rows) {
        isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                               tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, true);
    } else {
        isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), true);
    }
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}

/**
 * Perform APR Isotropic Convolution Operation on the GPU with a 5x5x5 kernel. Assumes that the access structures
 * and data are already on the GPU (tree_data_gpu and output_gpu need not be initialized, but must be allocated).
 *
 * @param access
 * @param tree_access
 * @param input_gpu
 * @param output_gpu
 * @param stencil_gpu
 * @param tree_data_gpu
 */
template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data_gpu, int* ne_rows_gpu, VectorData<int>& ne_counter, stencilType pad_value){

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
                                   level,
                                   ne_rows_gpu + offset,
                                   pad_value);

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
                                  ne_rows_gpu + offset,
                                  pad_value);
        }

//        error_check( cudaDeviceSynchronize() )
//        error_check( cudaPeekAtLastError() )
    }

//    error_check( cudaDeviceSynchronize() )
}


/**
 * Perform APR Isotropic Convolution Operation on the GPU with a 5x5x5 kernel. Initializes the necessary memory on
 * the GPU and copies the result back to the host after computation.
 *
 * @param access
 * @param tree_access
 * @param input
 * @param output
 * @param stencil
 * @param tree_data
 * @return
 */
template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                               VectorData<outputType>& output, VectorData<stencilType>& stencil, VectorData<treeType>& tree_data){

    APRTimer timer(false);
    timings ret;

    timer.start_timer("initialize GPU access (apr and tree)");
    tree_access.init_gpu();
    access.init_gpu(tree_access);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.init_access = timer.timings.back();

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 125);

    timer.start_timer("host data resize");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();

    timer.start_timer("compute nonempty rows");
    VectorData<int> ne_rows_ds; //non empty rows
    VectorData<int> ne_counter_ds; //non empty rows
    compute_ne_rows(tree_access, ne_counter_ds, ne_rows_ds);

    VectorData<int> ne_rows;
    VectorData<int> ne_counter;
    compute_ne_rows_new(access, ne_counter, ne_rows, 4);
    timer.stop_timer();
    ret.compute_ne_rows = timer.timings.back();

    timer.start_timer("allocate GPU memory");
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu(ne_rows_ds.data(), ne_rows_ds.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu(ne_rows.data(), ne_rows.size());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.allocation = timer.timings.back();

    timer.start_timer("transfer H2D");
    ne_rows_ds_gpu.copyH2D();
    ne_rows_gpu.copyH2D();
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("downsample");
    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.fill_tree = timer.timings.back();

    timer.start_timer("convolution");
    isotropic_convolve_555(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


/**
 * Perform APR Isotropic Convolution Operation on the GPU with a 5x5x5 kernel. Assumes that the access structures
 * and data are already on the GPU (tree_data_gpu and output_gpu need not be initialized, but must be allocated).
 *
 * @param access
 * @param tree_access
 * @param input_gpu
 * @param output_gpu
 * @param stencil_gpu
 * @param tree_data_gpu
 */
template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu, outputType* output_gpu,
                            stencilType* stencil_gpu, treeType* tree_data_gpu, stencilType pad_value){

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
                                           pad_value);

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
                                          pad_value);
        }

//        error_check( cudaDeviceSynchronize() )
//        error_check( cudaPeekAtLastError() )
    }

//    error_check( cudaDeviceSynchronize() )
}


/**
 * Perform APR Isotropic Convolution Operation on the GPU with a 5x5x5 kernel. Initializes the necessary memory on
 * the GPU and copies the result back to the host after computation.
 *
 * @param access
 * @param tree_access
 * @param input
 * @param output
 * @param stencil
 * @param tree_data
 * @return
 */
template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555_alt(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                    VectorData<outputType>& output, VectorData<stencilType>& stencil, VectorData<treeType>& tree_data){

    APRTimer timer(false);
    APRTimer timer2(false);

    timings ret;

    timer.start_timer("initialize GPU access (apr and tree)");
    tree_access.init_gpu();
    access.init_gpu(tree_access);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.init_access = timer.timings.back();

    assert(input.size() == access.total_number_particles());
    assert(stencil.size() == 125);

    timer.start_timer("host data resize");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();
    // add this to allocation time?

    timer.start_timer("allocate GPU memory");
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.allocation = timer.timings.back();

    timer.start_timer("transfer H2D");
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// fill tree by avg downsampling
    timer.start_timer("downsample");
    downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.fill_tree = timer.timings.back();

    timer.start_timer("convolution");
    isotropic_convolve_555(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.run_kernels = timer.timings.back();

    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                               outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu, stencilType pad_value){

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
                                           pad_value);

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
                                          level,
                                          pad_value);

            stencil_offset += 27;
        }
    }
}


template<typename inputType, typename outputType, typename stencilType, typename treeType>
void isotropic_convolve_555_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input_gpu,
                               outputType* output_gpu, stencilType* stencil_gpu, treeType* tree_data_gpu,
                               int* ne_rows_555, VectorData<int>& ne_counter_555, int* ne_rows_333, VectorData<int>& ne_counter_333, stencilType pad_value){

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
                                           ne_rows_555 + offset,
                                           pad_value);

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
                                          ne_rows_333 + offset,
                                          pad_value);

            stencil_offset += 27;
        }
    }
}



template<typename inputType, typename outputType, typename stencilType, typename treeType>
timings isotropic_convolve_555_ds(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                               VectorData<outputType>& output, PixelData<stencilType>& stencil, VectorData<treeType>& tree_data,
                               bool use_ne_rows, bool normalize_stencil){
    /*
     *  Perform APR Isotropic Convolution Operation on the GPU with a 5x5x5 kernel
     *
     *  conv_stencil needs to have 125 entries
     */

    APRTimer timer(false);
    timings ret;

    timer.start_timer("initialize GPU access (apr and tree)");
    tree_access.init_gpu();
    access.init_gpu(tree_access);
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.init_access = timer.timings.back();

    assert(input.size() == access.total_number_particles());
    assert(stencil.mesh.size() == 125);

    timer.start_timer("host data resize");
    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());
    timer.stop_timer();

    timer.start_timer("downsample stencil");
    VectorData<stencilType> stencil_vec;
    get_downsampled_stencils(stencil, stencil_vec, access.level_max() - access.level_min(), normalize_stencil);
//    const int stencil_size = 125 + (access.level_max() - access.level_min() - 1) * 27;
//
//    VectorData<stencilType> stencil_vec;
//    stencil_vec.resize(stencil_size);
//
//    for(int i = 0; i < 125; ++i) {
//        stencil_vec[i] = stencil.mesh[i];
//    }
//
//    int c = 125;
//    PixelData<stencilType> stencil_ds;
//    for (int level = access.level_max() - 1; level > access.level_min(); --level) {
//        downsample_stencil_new(stencil, stencil_ds, access.level_max() - level, normalize_stencil);
//        for(int i = 0; i < 27; ++i) {
//            stencil_vec[c+i] = stencil_ds.mesh[i];
//        }
//        c += 27;
//    }
    timer.stop_timer();
    ret.downsample_stencil = timer.timings.back();

    VectorData<int> ne_rows_ds;
    VectorData<int> ne_counter_ds;
    VectorData<int> ne_rows_333;
    VectorData<int> ne_counter_333;
    VectorData<int> ne_rows_555;
    VectorData<int> ne_counter_555;

    if(use_ne_rows) {
        timer.start_timer("compute ne rows");
        compute_ne_rows(tree_access, ne_counter_ds, ne_rows_ds);
        compute_ne_rows_new(access, ne_counter_333, ne_rows_333, 2);
        compute_ne_rows_new(access, ne_counter_555, ne_rows_555, 4);
        timer.stop_timer();
        //ret.counter_ne_rows = ne_rows.size();
        ret.compute_ne_rows = timer.timings.back();
    }

    timer.start_timer("allocate GPU memory");
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu(ne_rows_ds.data(), ne_rows_ds.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_333_gpu(ne_rows_333.data(), ne_rows_333.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_555_gpu(ne_rows_555.data(), ne_rows_555.size());
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<treeType*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<outputType*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> stencil_gpu(stencil_vec.data(), stencil_vec.size());
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.allocation = timer.timings.back();

    timer.start_timer("transfer H2D");
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();
    if(use_ne_rows) {
        ne_rows_ds_gpu.copyH2D();
        ne_rows_333_gpu.copyH2D();
        ne_rows_555_gpu.copyH2D();
    }
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_H2D = timer.timings.back();

    /// Fill the APR Tree by average downsampling
    timer.start_timer("fill tree");
    if(use_ne_rows) {
        downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    } else {
        downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    }
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.fill_tree = timer.timings.back();

    timer.start_timer("run kernels");
    if(use_ne_rows) {
        isotropic_convolve_555_ds(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, ne_rows_333_gpu.get(), ne_counter_333);
    } else {
        isotropic_convolve_555_ds(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
    }
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();

    ret.run_kernels = timer.timings.back();

    error_check( cudaDeviceSynchronize() )
    /// transfer the results back to the host
    timer.start_timer("transfer D2H");
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
    timer.stop_timer();
    ret.transfer_D2H = timer.timings.back();

    return ret;
}


template<typename inputType, typename stencilType>
void richardson_lucy(GPUAccessHelper& access, GPUAccessHelper& tree_access, VectorData<inputType>& input,
                     VectorData<stencilType>& output, PixelData<stencilType>& psf, int niter, bool downsample_stencil, bool normalize_stencil) {

    tree_access.init_gpu();
    access.init_gpu(tree_access);
    error_check( cudaDeviceSynchronize() )

    if(output.size() != input.size()) {
        output.resize(input.size());
    }

    /// allocate GPU memory
    ScopedCudaMemHandler<inputType*, JUST_ALLOC> input_gpu(input.data(), input.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> output_gpu(output.data(), output.size());

    /// copy data to GPU
    input_gpu.copyH2D();

    error_check( cudaDeviceSynchronize() )
    richardson_lucy(access, tree_access, input_gpu.get(), output_gpu.get(), psf, niter, downsample_stencil, normalize_stencil);
    error_check( cudaDeviceSynchronize() )

    /// copy result back to host
    output_gpu.copyD2H();
    error_check( cudaDeviceSynchronize() )
}


template<typename inputType, typename stencilType>
void richardson_lucy(GPUAccessHelper& access, GPUAccessHelper& tree_access, inputType* input, stencilType* output, PixelData<stencilType>& psf, int niter, bool downsample_stencil, bool normalize_stencil) {

    PixelData<stencilType> psf_flipped(psf, false);
    for(int i = 0; i < psf.mesh.size(); ++i) {
        psf_flipped.mesh[i] = psf.mesh[psf.mesh.size()-1-i];
    }

    VectorData<stencilType> psf_vec;
    VectorData<stencilType> psf_flipped_vec;

    if(downsample_stencil) {
        get_downsampled_stencils(psf, psf_vec, access.level_max() - access.level_min(), normalize_stencil);
        get_downsampled_stencils(psf_flipped, psf_flipped_vec, access.level_max() - access.level_min(), normalize_stencil);
    } else {
        psf_vec.resize(psf.mesh.size());
        psf_flipped_vec.resize(psf_flipped.mesh.size());

        std::copy(psf.mesh.begin(), psf.mesh.end(), psf_vec.begin());
        std::copy(psf_flipped.mesh.begin(), psf_flipped.mesh.end(), psf_flipped_vec.begin());
    }

    int kernel_size;
    if(psf.mesh.size() == 27) {
        kernel_size = 3;
    } else if(psf.mesh.size() == 125) {
        kernel_size = 5;
    } else {
        throw std::runtime_error("richardson_lucy is only implemented for 3x3x3 and 5x5x5 kernels");
    }

    VectorData<int> ne_rows_ds;
    VectorData<int> ne_counter_ds;
    VectorData<int> ne_rows_333;
    VectorData<int> ne_counter_333;
    VectorData<int> ne_rows_555;
    VectorData<int> ne_counter_555;

    /// non-empty rows precalculation (should always be worth it as they can be reused in all iterations)
    compute_ne_rows(tree_access, ne_counter_ds, ne_rows_ds);
    if(kernel_size == 3 || downsample_stencil) {
        compute_ne_rows_new(access, ne_counter_333, ne_rows_333, 2);
    }
    if(kernel_size == 5) {
        compute_ne_rows_new(access, ne_counter_555, ne_rows_555, 4);
    }

    /// allocate GPU memory
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> relative_blur(NULL, access.total_number_particles());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> error_est(NULL, access.total_number_particles());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> tree_data_gpu(NULL, tree_access.total_number_particles());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> psf_vec_gpu(psf_vec.data(), psf_vec.size());
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> psf_flipped_vec_gpu(psf_flipped_vec.data(), psf_flipped_vec.size());

    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu(ne_rows_ds.data(), ne_rows_ds.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_333_gpu(ne_rows_333.data(), ne_rows_333.size());
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_555_gpu(ne_rows_555.data(), ne_rows_555.size());


    psf_vec_gpu.copyH2D();
    psf_flipped_vec_gpu.copyH2D();
    ne_rows_ds_gpu.copyH2D();
    if(kernel_size == 3 || downsample_stencil) {
        ne_rows_333_gpu.copyH2D();
    }
    if(kernel_size == 5) {
        ne_rows_555_gpu.copyH2D();
    }

    /// set block size for cuda kernels
    const size_t numParts = access.total_number_particles();

    dim3 grid_dim(8, 1, 1);
    dim3 block_dim(256, 1, 1);

    /// initialize output as the input image
    fillWithValue<<< grid_dim, block_dim >>> (output, (stencilType)1, numParts);

    stencilType average_h;
    stencilType* average_d;
    error_check( cudaMalloc(&average_d, sizeof(stencilType)) )

    error_check( cudaDeviceSynchronize() )

    for(int i = 0; i < niter; ++i) {

        downsample_avg(access, tree_access, output, tree_data_gpu.get());
        compute_average<512><<<1, 512>>>(output, average_d, 4096);

        error_check( cudaDeviceSynchronize() )
        error_check( cudaMemcpy(&average_h, average_d, sizeof(stencilType), cudaMemcpyDeviceToHost) )// todo: find a clean way to avoid this...
        error_check( cudaDeviceSynchronize() )

        if(kernel_size == 5) {
            if(downsample_stencil) {
                isotropic_convolve_555_ds(access, tree_access, output, relative_blur.get(), psf_vec_gpu.get(),
                                          tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, ne_rows_333_gpu.get(), ne_counter_333, average_h);
            } else {
                isotropic_convolve_555(access, tree_access, output, relative_blur.get(), psf_vec_gpu.get(),
                                       tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, average_h);
            }
        } else {
            isotropic_convolve_333(access, tree_access, output, relative_blur.get(), psf_vec_gpu.get(),
                                   tree_data_gpu.get(), ne_rows_333_gpu.get(), ne_counter_333, downsample_stencil, average_h);
        }

        error_check( cudaDeviceSynchronize() )

        elementWiseDiv<<< grid_dim, block_dim >>> (input, relative_blur.get(), relative_blur.get(), numParts);

        error_check( cudaDeviceSynchronize() )

        downsample_avg(access, tree_access, relative_blur.get(), tree_data_gpu.get());

        compute_average<512><<<1, 512>>>(relative_blur.get(), average_d, 4096);

        error_check( cudaDeviceSynchronize() )
        error_check( cudaMemcpy(&average_h, average_d, sizeof(stencilType), cudaMemcpyDeviceToHost) ) // todo: find a clean way to avoid this...
        error_check( cudaDeviceSynchronize() )

        if(kernel_size == 5) {
            if(downsample_stencil) {
                isotropic_convolve_555_ds(access, tree_access, relative_blur.get(), error_est.get(), psf_flipped_vec_gpu.get(),
                                          tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, ne_rows_333_gpu.get(), ne_counter_333, average_h);
            } else {
                isotropic_convolve_555(access, tree_access, relative_blur.get(), error_est.get(), psf_flipped_vec_gpu.get(),
                                       tree_data_gpu.get(), ne_rows_555_gpu.get(), ne_counter_555, average_h);
            }
        } else {
            isotropic_convolve_333(access, tree_access, relative_blur.get(), error_est.get(), psf_flipped_vec_gpu.get(),
                                       tree_data_gpu.get(), ne_rows_333_gpu.get(), ne_counter_333, downsample_stencil, average_h);
        }

        error_check( cudaDeviceSynchronize() )

        elementWiseMult<<< grid_dim, block_dim >>> (output, error_est.get(), numParts);

        error_check( cudaDeviceSynchronize() )
    }
}


template<typename inputType, typename stencilType>
void richardson_lucy_pixel(inputType* input, stencilType* output, stencilType* psf, stencilType* psf_flipped, int kernel_size, int npixels, int niter, std::vector<int>& dims) {

    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> relative_blur(NULL, npixels);
    ScopedCudaMemHandler<stencilType*, JUST_ALLOC> error_est(NULL, npixels);

    dim3 grid_dim(8, 1, 1);
    dim3 block_dim(256, 1, 1);

    const int chunkSize = (kernel_size == 3) ? 32 : 16;
    const int convBlockSize = (kernel_size == 3) ? 4 : 8;

    const int x_blocks = (kernel_size == 3) ? (dims[1] + convBlockSize - 3) / (convBlockSize-2) : (dims[1] + convBlockSize - 5) / (convBlockSize-4);
    const int z_blocks = (kernel_size == 3) ? (dims[0] + convBlockSize - 3) / (convBlockSize-2) : (dims[0] + convBlockSize - 5) / (convBlockSize-4);

    dim3 conv_grid_dim(x_blocks, 1, z_blocks);
    dim3 conv_block_dim(chunkSize, convBlockSize, convBlockSize);

    /// initialize output with 1s
    fillWithValue<<< grid_dim, block_dim >>> (output, (stencilType)1, npixels);
    error_check( cudaDeviceSynchronize() )

    for(int i = 0; i < niter; ++i) {

        if(kernel_size == 5) {
            conv_pixel_555_chunked<16, 8><<< conv_grid_dim, conv_block_dim >>> (output, relative_blur.get(), psf, dims[0], dims[1], dims[2]);
        } else {
            conv_pixel_333_chunked<32, 4><<< conv_grid_dim, conv_block_dim >>> (output, relative_blur.get(), psf, dims[0], dims[1], dims[2]);
        }
        error_check( cudaDeviceSynchronize() )

        elementWiseDiv<< < grid_dim, block_dim >> > (input, relative_blur.get(), relative_blur.get(), npixels);
        error_check( cudaDeviceSynchronize() )

        if(kernel_size == 5) {
            conv_pixel_555_chunked<16, 8><<< conv_grid_dim, conv_block_dim >>> (relative_blur.get(), error_est.get(), psf_flipped, dims[0], dims[1], dims[2]);
        } else {
            conv_pixel_333_chunked<32, 4><<< conv_grid_dim, conv_block_dim >>> (relative_blur.get(), error_est.get(), psf_flipped, dims[0], dims[1], dims[2]);
        }
        error_check( cudaDeviceSynchronize() )

        elementWiseMult<<< grid_dim, block_dim >>> (output, error_est.get(), npixels);
        error_check( cudaDeviceSynchronize() )
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


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_chunked(const uint64_t* level_xz_vec,
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
                                     const int level,
                                     const int* offset_ind,
                                     const stencilType pad_value) {

    const int index = offset_ind[blockIdx.x];

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {

        // out of bounds --> zero pad and return
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
                y_0 = y_num*2;
            }
        }
        __syncthreads();

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

        //__syncthreads();
        if(y_0 <= (y_chunk+1)*chunkSizeInternal) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % chunkSize] = f_0;
        }

        //__syncthreads();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num)) {
            local_patch[threadIdx.z][threadIdx.y][(y_p+1) % chunkSize] = f_p;
        }

        __syncthreads();

        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 1, neighbour_sum)

        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_chunked(const uint64_t* level_xz_vec,
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
                                          const int level,
                                          const int* offset_ind,
                                          const stencilType pad_value) {

    const int index = offset_ind[blockIdx.x];

    const int x_index = index % x_num + threadIdx.y - 1;
    const int z_index = index / x_num + threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {
        // out of bounds --> zero pad and return
        return;
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

        __syncthreads();
        while( y_t < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_t++;

            if( (sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row] ) {
                y_t = y_vec_tree[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
                f_t = tree_data[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
            } else {
                y_t = INT32_MAX;
            }
        }

        __syncthreads();
        while( y_p < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncthreads();
        if( y_0 <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % N] = f_0;
        }
        __syncthreads();
        if( y_t <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+1) % N] = f_t;
        }
        __syncthreads();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 1) % N] = f_p;
        }

        __syncthreads();
        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {
            float neigh_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0+1, neigh_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;
    } // end for y_chunk
}


//// without non-empty rows precomputation

template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_max_333_chunked(const uint64_t* level_xz_vec,
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
                                     const int level,
                                     const stencilType pad_value) {

    const int z_index = blockIdx.z * (blockSize-2) + threadIdx.z - 1;
    const int x_index = blockIdx.x * (blockSize-2) + threadIdx.y - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {

        // out of bounds --> return
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

        __syncthreads();

        while( y_p < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(chunkSize/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncthreads();
        if( y_0 <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % chunkSize] = f_0;
        }

        __syncthreads();
        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p+1) % chunkSize] = f_p;
        }

        __syncthreads();

        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {
            float neighbour_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 1, neighbour_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void conv_interior_333_chunked(const uint64_t* level_xz_vec,
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
                                          const int level,
                                          const stencilType pad_value) {

    const int z_index = blockIdx.z * (blockSize-2) + threadIdx.z - 1;
    const int x_index = blockIdx.x * (blockSize-2) + threadIdx.y - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {
        // out of bounds --> zero pad and return
        return;
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

        __syncthreads();
        while( y_t < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_t++;

            if( (sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row] ) {
                y_t = y_vec_tree[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
                f_t = tree_data[sparse_block_t*chunkSize + global_index_begin_t_s[row] + threadIdx.x];
            } else {
                y_t = INT32_MAX;
            }
        }

        __syncthreads();
        while( y_p < (y_chunk*chunkSizeInternal - 1) ) {
            sparse_block_p++;

            if( (global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + sparse_block_p*(chunkSize/2) + threadIdx.x/2];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncthreads();

        if( y_0 <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % N] = f_0;
        }

        __syncthreads();
        if( y_t <= (y_chunk+1)*chunkSizeInternal ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+1) % N] = f_t;
        }
        __syncthreads();

        if( (y_p <= (y_chunk+1)*chunkSizeInternal) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 1) % N] = f_p;
        }

        __syncthreads();


        if( (y_0 >= y_chunk*chunkSizeInternal) && (y_0 < (y_chunk+1)*chunkSizeInternal) ) {
            float neigh_sum = 0;
            LOCALPATCHCONV333_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0+1, neigh_sum)

        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;
    } // end for y_chunk
}




template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
L(1024, 2)
conv_max_555_chunked(const uint64_t* level_xz_vec,
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
                     const int level,
                     const int* offset_ind,
                     const stencilType pad_value) {

    const int index = offset_ind[blockIdx.x];

    const int x_index = index % x_num + threadIdx.y - 2;
    const int z_index = index / x_num + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[5][5][5];

    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

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

    __syncthreads();
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

        __syncthreads();
        while( (y_p < (y_chunk*chunkSizeInternal - 2)) ) {
            sparse_block_p++;
            if( (global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncthreads();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0 + 2) % N] = f_0;
        }

        __syncthreads();
        if( (y_p <= ((y_chunk+1)*chunkSizeInternal+1)) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 2) % N] = f_p;
        }

        __syncthreads();

        if( ((y_0 >= (y_chunk*chunkSizeInternal)) && (y_0 < ((y_chunk+1)*chunkSizeInternal))) ) {
            float neighbour_sum = 0;
            LOCALPATCHCONV555_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 2, neighbour_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void
L(1024, 2)
conv_interior_555_chunked(const uint64_t* level_xz_vec,
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
                         const int level,
                         const int* offset_ind,
                         const stencilType pad_value) {

    const int index = offset_ind[blockIdx.x];

    const int x_index = index % x_num + threadIdx.y - 2;
    const int z_index = index / x_num + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    /// copy the stencil to shared memory
    __shared__ stencilType local_stencil[5][5][5];
    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }

    /// initialize "local isotropic patch" buffer in shared memory
    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value; // zero pad

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

    __syncthreads();

    if( ((global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row]) ) {
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
        __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+2) % N] = f_0;
        }

        // insert tree particle
        __syncthreads();
        if( y_t <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+2) % N] = f_t;
        }

        // insert parent particle
        __syncthreads();
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
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;
    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
L(1024, 2)
conv_max_555_chunked(const uint64_t* level_xz_vec,
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
                     const int level,
                     const stencilType pad_value) {

    const int x_index = blockIdx.x * (blockSize - 4) + threadIdx.y - 2;
    const int z_index = blockIdx.z * (blockSize - 4) + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[5][5][5];

    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

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

    __syncthreads();
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

        __syncthreads();
        while( (y_p < (y_chunk*chunkSizeInternal - 2)) ) {
            sparse_block_p++;
            if( (global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)) < global_index_end_p_s[row] ) {
                y_p = 2*y_vec[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)] + y_offset_p;
                f_p = input_particles[global_index_begin_p_s[row] + (sparse_block_p*(N/2) + threadIdx.x/2)];
            } else{
                y_p = INT32_MAX;
            }
        }

        __syncthreads();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0 + 2) % N] = f_0;
        }

        __syncthreads();
        if( (y_p <= ((y_chunk+1)*chunkSizeInternal+1)) && (y_p < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_p + 2) % N] = f_p;
        }

        __syncthreads();

        if( ((y_0 >= (y_chunk*chunkSizeInternal)) && (y_0 < ((y_chunk+1)*chunkSizeInternal))) ) {
            float neighbour_sum = 0;
            LOCALPATCHCONV555_N(output_particles, update_index, threadIdx.z, threadIdx.y, y_0 + 2, neighbour_sum)
        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;

    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType, typename treeType>
__global__ void
L(1024, 2)
conv_interior_555_chunked(const uint64_t* level_xz_vec,
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
                          const int level,
                          const stencilType pad_value) {

    const int x_index = blockIdx.x * (blockSize - 4) + threadIdx.y - 2;
    const int z_index = blockIdx.z * (blockSize - 4) + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    /// copy the stencil to shared memory
    __shared__ stencilType local_stencil[5][5][5];
    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }

    /// initialize "local isotropic patch" buffer in shared memory
    __shared__ stencilType local_patch[blockSize][blockSize][N];
    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value; // zero pad

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

    __syncthreads();

    if( ((global_index_begin_t_s[row] + threadIdx.x) < global_index_end_t_s[row]) ) {
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
        __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
        if( y_0 <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+2) % N] = f_0;
        }

        // insert tree particle
        __syncthreads();
        if( y_t <= ((y_chunk+1)*chunkSizeInternal+1) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_t+2) % N] = f_t;
        }

        // insert parent particle
        __syncthreads();
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
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = pad_value;
    } // end for y_chunk
}


template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void conv_pixel_333_chunked(const inputType* input_image,
                                       outputType* output_image,
                                       const stencilType* stencil,
                                       const int z_num,
                                       const int x_num,
                                       const int y_num) {

    const int block_dim = 4;

    const int x_index = blockIdx.x * 2 + threadIdx.y - 1;
    const int z_index = blockIdx.z * 2 + threadIdx.z - 1;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[3][3][3];

    if((threadIdx.y < 3) && (threadIdx.x < 3) && (threadIdx.z < 3)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 9 + threadIdx.x * 3 + threadIdx.y];
    }

    __syncthreads();

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {

        // out of bounds --> return
        return;
    }

    const bool not_ghost = (threadIdx.y > 0) && (threadIdx.y < (block_dim - 1)) &&
                           (threadIdx.z > 0) && (threadIdx.z < (block_dim - 1));
    //(threadIdx.y > 0) && (threadIdx.y < (blockDim.y - 1));


    const size_t row_begin = x_index * y_num + z_index * x_num * y_num;

    __syncthreads();

    int y_0;

    // overlapping y chunks

    const int chunkSizeInternal = chunkSize-2;
    const int number_y_chunks = (y_num + chunkSizeInternal - 1) / chunkSizeInternal;

    __syncthreads();

    for(int y_chunk = 0; y_chunk < number_y_chunks; ++y_chunk) {

        __syncthreads();

        y_0 = y_chunk * chunkSizeInternal - 1 + threadIdx.x;

        if( (y_0 >= 0) && (y_0 < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+1) % N] = input_image[row_begin + y_0];
        }

        __syncthreads();

        if( (y_0 >= (y_chunk*(chunkSizeInternal))) && (y_0 < min(((y_chunk+1)*(chunkSizeInternal)), y_num)) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV333_N(output_image, (row_begin+y_0), threadIdx.z, threadIdx.y, y_0 + 1, neighbour_sum)

        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    } // end for y_chunk
}



template<unsigned int chunkSize, unsigned int blockSize, typename inputType, typename outputType, typename stencilType>
__global__ void
L(1024, 2)
conv_pixel_555_chunked(const inputType* input_image,
                                       outputType* output_image,
                                       const stencilType* stencil,
                                       const int z_num,
                                       const int x_num,
                                       const int y_num) {

    //const int block_dim = 4;

    const int x_index = blockIdx.x * (blockSize-4) + threadIdx.y - 2;
    const int z_index = blockIdx.z * (blockSize-4) + threadIdx.z - 2;

    const unsigned int N = chunkSize;

    __shared__ stencilType local_stencil[5][5][5];

    if((threadIdx.y < 5) && (threadIdx.x < 5) && (threadIdx.z < 5)){
        local_stencil[threadIdx.z][threadIdx.x][threadIdx.y] = stencil[threadIdx.z * 25 + threadIdx.x * 5 + threadIdx.y];
    }

    __syncthreads();

    __shared__ stencilType local_patch[blockSize][blockSize][N];

    local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    if( (x_index < 0) || (x_index >= x_num) || (z_index < 0) || (z_index >= z_num) ) {

        // out of bounds --> return
        return;
    }

    const bool not_ghost = (threadIdx.y > 1) && (threadIdx.y < (blockSize - 2)) &&
                           (threadIdx.z > 1) && (threadIdx.z < (blockSize - 2));

    const size_t row_begin = x_index * y_num + z_index * x_num * y_num;

    __syncthreads();

    int y_0;

    // overlapping y chunks

    const int chunkSizeInternal = chunkSize-4;
    const int number_y_chunks = (y_num + chunkSizeInternal - 1) / chunkSizeInternal;

    __syncthreads();

    for(int y_chunk = 0; y_chunk < number_y_chunks; ++y_chunk) {

        __syncthreads();

        y_0 = y_chunk * chunkSizeInternal - 2 + threadIdx.x;

        if( (y_0 >= 0) && (y_0 < y_num) ) {
            local_patch[threadIdx.z][threadIdx.y][(y_0+2) % N] = input_image[row_begin + y_0];
        }

        __syncthreads();

        if( (y_0 >= (y_chunk*(chunkSizeInternal))) && (y_0 < min(((y_chunk+1)*(chunkSizeInternal)), y_num)) ) {

            float neighbour_sum = 0;
            LOCALPATCHCONV555_N(output_image, (row_begin+y_0), threadIdx.z, threadIdx.y, (y_0+2), neighbour_sum)

        }

        __syncthreads();
        local_patch[threadIdx.z][threadIdx.y][threadIdx.x] = 0;

    } // end for y_chunk
}


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


template<typename T>
__global__ void copyKernel(const T* in, T* out, const size_t size){

    for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        out[idx] = in[idx];
    }
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
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<float>&);
template timings isotropic_convolve_333(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<double>&);

//apr 333 without ne rows
template timings isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<float>&);
template timings isotropic_convolve_333_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<double>&);

//apr 555
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<float>&);
template timings isotropic_convolve_555(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<double>&);

//apr 555 without ne rows
template timings isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<float>&, VectorData<float>&);
template timings isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<float>&);
template timings isotropic_convolve_555_alt(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, VectorData<double>&, VectorData<double>&);

//apr 333 with downsample stencil
template timings isotropic_convolve_333_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, PixelData<float>&, VectorData<float>&, bool, bool);
template timings isotropic_convolve_333_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, PixelData<float>&, VectorData<float>&, bool, bool);
template timings isotropic_convolve_333_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, PixelData<float>&, VectorData<float>&, bool, bool);
template timings isotropic_convolve_333_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, PixelData<double>&, VectorData<float>&, bool, bool);
template timings isotropic_convolve_333_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<double>&, PixelData<double>&, VectorData<double>&, bool, bool);

//apr 555 with downsample stencil
template timings isotropic_convolve_555_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&,VectorData<float>&, PixelData<float>&, VectorData<float>&, bool, bool);
template timings isotropic_convolve_555_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&,VectorData<float>&, PixelData<float>&, VectorData<float>&, bool, bool);
template timings isotropic_convolve_555_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&,VectorData<double>&, PixelData<float>&, VectorData<float>&, bool, bool);
template timings isotropic_convolve_555_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&,VectorData<double>&, PixelData<double>&, VectorData<float>&, bool, bool);
template timings isotropic_convolve_555_ds(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&,VectorData<double>&, PixelData<double>&, VectorData<double>&, bool, bool);

//

/// deconv
template void richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, VectorData<float>&, VectorData<float>&, PixelData<float>&, int, bool, bool);
template void richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, VectorData<uint16_t>&, VectorData<float>&, PixelData<float>&, int, bool, bool);

//template void richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, VectorData<double>&, VectorData<double>&, VectorData<double>&, int, bool);
template void richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, float*, float*, PixelData<float>&, int, bool, bool);
template void richardson_lucy(GPUAccessHelper&, GPUAccessHelper&, uint16_t*, float*, PixelData<float>&, int, bool, bool);

template void richardson_lucy_pixel(float*, float*, float*, float*, int, int, int, std::vector<int>&);
template void richardson_lucy_pixel(uint16_t*, float*, float*, float*, int, int, int, std::vector<int>&);


