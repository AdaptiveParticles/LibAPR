#include <algorithm>
#include <vector>
#include <array>
#include <iostream>
#include <cassert>
#include <limits>
#include <chrono>
#include <iomanip>

#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/ExtraParticleData.hpp"

#include "thrust/device_vector.h"
#include "thrust/tuple.h"
#include "thrust/copy.h"

struct cmdLineOptions{
    std::string output = "output";
    std::string stats = "";
    std::string directory = "";
    std::string input = "";
};

bool command_option_exists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

cmdLineOptions read_command_line_options(int argc, char **argv) {
    cmdLineOptions result;

    if(argc == 1) {
        std::cerr << "Usage: \"Example_apr_neighbour_access -i input_apr_file -d directory\"" << std::endl;
        exit(1);
    }
    if(command_option_exists(argv, argv + argc, "-i")) {
        result.input = std::string(get_command_option(argv, argv + argc, "-i"));
    } else {
        std::cout << "Input file required" << std::endl;
        exit(2);
    }
    if(command_option_exists(argv, argv + argc, "-d")) {
        result.directory = std::string(get_command_option(argv, argv + argc, "-d"));
    }
    if(command_option_exists(argv, argv + argc, "-o")) {
        result.output = std::string(get_command_option(argv, argv + argc, "-o"));
    }

    return result;
}

__global__ void insert(
    std::size_t _level,
    std::size_t _z_index,
    const thrust::tuple<std::size_t,std::size_t>* _line_offsets,
    const std::uint16_t*           _y_ex,
    const std::uint16_t*           _pdata,
    const std::size_t*             _offsets,
    std::size_t                    _max_y,
    std::size_t                    _max_x,
    std::size_t                    _nparticles,
    std::uint16_t*                 _temp_vec,
    std::size_t                    _stencil_size
    ){

    unsigned int y_index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int x_index = blockDim.y * blockIdx.y + threadIdx.y;

    if(x_index >= _max_x){
        return; // out of bounds
    }

    if(y_index >= _max_y){
        return; // out of bounds
    }

    auto level_zx_offset = _offsets[_level] + _max_x * _z_index + x_index;
    auto row_start = _line_offsets[level_zx_offset];

    if(thrust::get<1>(row_start) == 0)
        return;

    auto particle_index_begin = thrust::get<0>(row_start);
    auto particle_index_end   = thrust::get<1>(row_start);

    auto t_index = x_index*_max_y + ((_z_index % _stencil_size)*_max_y*_max_x) ;

    for (std::size_t global_index = particle_index_begin;
         global_index <= particle_index_end; ++global_index) {

        uint16_t current_particle_value = _pdata[global_index];
        auto y = _y_ex[global_index];
        _temp_vec[t_index+y] = current_particle_value;

    }


}

__global__ void push_back(
    std::size_t _level,
    std::size_t _z_index,
    const thrust::tuple<std::size_t,std::size_t>* _line_offsets,
    const std::uint16_t*           _y_ex,
    const std::uint16_t*           _temp_vec,
    const std::size_t*             _offsets,
    std::size_t                    _max_y,
    std::size_t                    _max_x,
    std::size_t                    _nparticles,
    std::uint16_t*                 _pdata,
    std::size_t                    _stencil_size
    ){

    unsigned int y_index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int x_index = blockDim.y * blockIdx.y + threadIdx.y;

    if(x_index >= _max_x){
        return; // out of bounds
    }

    if(y_index >= _max_y){
        return; // out of bounds
    }

    auto level_zx_offset = _offsets[_level] + _max_x * _z_index + x_index;
    auto row_start = _line_offsets[level_zx_offset];

    if(thrust::get<1>(row_start) == 0)
        return;

    auto particle_index_begin = thrust::get<0>(row_start);
    auto particle_index_end   = thrust::get<1>(row_start);

    auto t_index = x_index*_max_y + ((_z_index % _stencil_size)*_max_y*_max_x) ;

    for (std::size_t global_index = particle_index_begin;
         global_index <= particle_index_end; ++global_index) {

        auto y = _y_ex[global_index];
        _pdata[global_index] = _temp_vec[t_index+y];


    }


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    // Read provided APR file
    cmdLineOptions options = read_command_line_options(argc, argv);
    const int reps = 20;

    std::string fileName = options.directory + options.input;
    APR<uint16_t> apr;
    apr.read_apr(fileName);

    // Get dense representation of APR
    APRIterator<uint16_t> aprIt(apr);

    ///////////////////////////
    ///
    /// Sparse Data for GPU
    ///
    ///////////////////////////

    std::vector<std::tuple<std::size_t,std::size_t>> level_zx_index_start;//size = number of rows on all levels
    std::vector<std::uint16_t> y_explicit;y_explicit.reserve(aprIt.total_number_particles());//size = number of particles
    std::vector<std::uint16_t> particle_values;particle_values.reserve(aprIt.total_number_particles());//size = number of particles
    std::vector<std::size_t> level_offset(aprIt.level_max()+1,UINT64_MAX);//size = number of levels

    std::size_t x = 0;
    std::size_t z = 0;

    std::size_t zx_counter = 0;


    for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
        level_offset[level] = zx_counter;

        for (z = 0; z < aprIt.spatial_index_z_max(level); ++z) {
            for (x = 0; x < aprIt.spatial_index_x_max(level); ++x) {

                zx_counter++;
                if (aprIt.set_new_lzx(level, z, x) < UINT64_MAX) {
                    level_zx_index_start.emplace_back(std::make_tuple<std::size_t,std::size_t>(aprIt.global_index(),
                                                                                               aprIt.particles_zx_end(level,z,x)-1)); //This stores the begining and end global index for each level_xz_row
                } else {
                    level_zx_index_start.emplace_back(std::make_tuple<std::size_t,std::size_t>(UINT64_MAX, 0)); //This stores the begining and end global index for each level_
                }

                for (aprIt.set_new_lzx(level, z, x);
                     aprIt.global_index() < aprIt.particles_zx_end(level, z,
                                                                   x); aprIt.set_iterator_to_particle_next_particle()) {
                    y_explicit.emplace_back(aprIt.y());
                    particle_values.emplace_back(apr.particles_intensities[aprIt]);

                }
            }

        }
    }

    // std::vector<uint16_t> cpu_access_data(apr.particles_intensities.data.size(),std::numeric_limits<std::uint16_t>::max());

    // for ( int r = 0;r<reps;++r){
    //     auto start_cpu = std::chrono::high_resolution_clock::now();


    //     for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {

    //         const int x_num = aprIt.spatial_index_x_max(level);
    //         //const int z_num = aprIt.spatial_index_z_max(level);

    //         for (z = 0; z < aprIt.spatial_index_z_max(level); ++z) {
    //             for (x = 0; x < aprIt.spatial_index_x_max(level); ++x) {
    //                 if(level_offset[level]<UINT64_MAX) {
    //                     uint64_t level_xz_offset = level_offset[level] + x_num * z + x;
    //                     if (std::get<1>(level_zx_index_start[level_xz_offset])) {
    //                         uint64_t particle_index_begin = std::get<0>(level_zx_index_start[level_xz_offset]);
    //                         uint64_t particle_index_end = std::get<1>(level_zx_index_start[level_xz_offset]);

    //                         for (uint64_t global_index = particle_index_begin;
    //                              global_index <= particle_index_end; ++global_index) {

    //                             uint16_t current_particle_value = particle_values[global_index];

    //                             cpu_access_data[global_index] = (current_particle_value);

    //                         }
    //                     }
    //                 }

    //             }
    //         }
    //     }

    //     auto end_cpu = std::chrono::high_resolution_clock::now();

    //     std::chrono::duration<double, std::milli> diff_cpu = end_cpu-start_cpu;
    //     std::cout << std::setw(3) << r << " CPU:      " << diff_cpu   .count() << " ms\n";

    // }

    ////////////////////
    ///
    /// Example of doing our level,z,x access using the GPU data structure
    ///
    /////////////////////
    auto start = std::chrono::high_resolution_clock::now();


    thrust::host_vector<thrust::tuple<std::size_t,std::size_t> > h_level_zx_index_start(level_zx_index_start.size());
    thrust::transform(level_zx_index_start.begin(), level_zx_index_start.end(),
                      h_level_zx_index_start.begin(),
                      [] ( const auto& _el ){
                          return thrust::make_tuple(std::get<0>(_el), std::get<1>(_el));
                      } );

    thrust::device_vector<thrust::tuple<std::size_t,std::size_t> > d_level_zx_index_start = h_level_zx_index_start;

    thrust::device_vector<std::uint16_t> d_y_explicit(y_explicit.begin(), y_explicit.end());
    thrust::device_vector<std::uint16_t> d_particle_values(particle_values.begin(), particle_values.end());
    thrust::device_vector<std::uint16_t> d_test_access_data(d_particle_values.size(),std::numeric_limits<std::uint16_t>::max());

    thrust::device_vector<std::size_t> d_level_offset(level_offset.begin(),level_offset.end());

    std::size_t max_elements = 0;
    const int stencil_size =5;
    for (int level = aprIt.level_min(); level <= aprIt.level_max(); ++level) {
        auto xtimesy = aprIt.spatial_index_y_max(level) + (stencil_size - 1);
        xtimesy *= aprIt.spatial_index_x_max(level) + (stencil_size - 1);
        if(max_elements < xtimesy)
            max_elements = xtimesy;
    }
    thrust::device_vector<std::uint16_t> d_temp_vec(max_elements*stencil_size,0);

    const thrust::tuple<std::size_t,std::size_t>* levels =  thrust::raw_pointer_cast(d_level_zx_index_start.data());
    const std::uint16_t*             y_ex   =  thrust::raw_pointer_cast(d_y_explicit.data());
    const std::uint16_t*             pdata  =  thrust::raw_pointer_cast(d_particle_values.data());
    const std::size_t*             offsets= thrust::raw_pointer_cast(d_level_offset.data());
    std::uint16_t*                   tvec = thrust::raw_pointer_cast(d_temp_vec.data());
    std::uint16_t*                   expected = thrust::raw_pointer_cast(d_test_access_data.data());

    if(cudaGetLastError()!=cudaSuccess){
        std::cerr << "memory transfers failed!\n";

    }
    auto end_gpu_tx = std::chrono::high_resolution_clock::now();

    for ( int r = 0;r<reps;++r){

        auto start_gpu_kernel = std::chrono::high_resolution_clock::now();

        for (int lvl = aprIt.level_min(); lvl <= aprIt.level_max(); ++lvl) {

            const int y_num = aprIt.spatial_index_y_max(lvl);
            const int x_num = aprIt.spatial_index_x_max(lvl);
            const int z_num = aprIt.spatial_index_z_max(lvl);

            dim3 threads(32,32);
            dim3 blocks((y_num + threads.x- 1)/threads.x,
                        (x_num + threads.y- 1)/threads.y
                );

            for(int z = 0;z<z_num;++z){

                insert<<<blocks,threads>>>(lvl,
                                           z,
                                           levels,
                                           y_ex,
                                           pdata,
                                           offsets,
                                           y_num,x_num,
                                           particle_values.size(),
                                           tvec,
                                           stencil_size);

                if(cudaGetLastError()!=cudaSuccess){
                    std::cerr << "on " << lvl << " the cuda kernel does not run!\n";
                    break;
                }

                push_back<<<blocks,threads>>>(lvl,
                                              z,
                                              levels,
                                              y_ex,
                                              tvec,
                                              offsets,
                                              y_num,x_num,
                                              particle_values.size(),
                                              expected,
                                              stencil_size);

            }
        }
        cudaDeviceSynchronize();
        auto end_gpu_kernel = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> rep_diff = end_gpu_kernel - start_gpu_kernel;
        std::cout << std::setw(3) << r << " GPU:      " << rep_diff  .count() << " ms\n";

    }

    auto end_gpu_kernels = std::chrono::high_resolution_clock::now();

    std::vector<std::uint16_t> test_access_data(d_test_access_data.size(),std::numeric_limits<std::uint16_t>::max());
    thrust::copy(d_test_access_data.begin(), d_test_access_data.end(), test_access_data.begin());

    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double,std::milli> gpu_tx_up = end_gpu_tx - start;
    std::chrono::duration<double,std::milli> gpu_tx_down = end_gpu - end_gpu_kernels;

    std::cout << "   GPU: up   " << gpu_tx_up  .count() << " ms\n";
    std::cout << "   GPU: down " << gpu_tx_down.count() << " ms\n";

    assert(test_access_data.back() != std::numeric_limits<std::uint16_t>::max());

    //////////////////////////
    ///
    /// Now check the data
    ///
    ////////////////////////////

    bool success = true;

    for (std::size_t i = 0; i < test_access_data.size(); ++i) {
        if(apr.particles_intensities.data[i]!=test_access_data[i]){
            success = false;
            std::cout << i << " expected: " << apr.particles_intensities.data[i] << ", received: " << test_access_data[i] << "\n";
            break;
        }
    }

    if(success){
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }


}
