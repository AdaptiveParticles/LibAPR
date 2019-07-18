//
// Created by Joel Jonsson on 2019-07-18.
//

#ifndef LIBAPR_FILTERBENCHMARKS_HPP
#define LIBAPR_FILTERBENCHMARKS_HPP

#include <algorithm>
#include <iostream>

#include "BenchAPRHelper.hpp"
#include "numerics/APRFilter.hpp"

#ifdef APR_USE_CUDA
#include "numerics/APRIsoConvGPU.hpp"
#endif

template<typename partsType>
inline void bench_apr_convolve(APR& apr,ParticleData<partsType>& parts, int num_rep,AnalysisData& analysisData,int stencil_size = 3);

template<typename partsType>
inline void bench_apr_convolve_pencil(APR& apr,ParticleData<partsType>& parts, int num_rep,AnalysisData& analysisData,int stencil_size = 3);

template<typename partsType>
inline void bench_pixel_convolve(APR& apr,ParticleData<partsType>& parts, int num_rep,AnalysisData& analysisData,int stencil_size = 3);

#ifdef APR_USE_CUDA
template<typename partsType>
inline void bench_apr_convolve_cuda(APR& apr,ParticleData<partsType>& parts, int num_rep,AnalysisData& analysisData,int stencil_size = 3);

template<typename partsType>
inline void bench_pixel_convolve_cuda(APR& apr,ParticleData<partsType>& parts, int num_rep,AnalysisData& analysisData,int stencil_size = 3);
#endif


template<typename partsType>
inline void bench_apr_convolve(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size){

    APRTimer timer(true);

    std::vector<PixelData<float>> stencils;
    stencils.resize(1);

    auto it = apr.iterator();

    if(it.number_dimensions() ==3){
        stencils[0].init(stencil_size, stencil_size, stencil_size);
    } else if (it.number_dimensions() ==2){
        stencils[0].init(stencil_size, stencil_size, 1);
    } else if (it.number_dimensions() ==1){
        stencils[0].init(stencil_size, 1, 1);
    }

    // unique stencil elements
    float sum = 0;
    for(int i = 0; i < (int) stencils[0].mesh.size(); ++i) {
        sum += i;
    }
    for(int i = 0; i < (int) stencils[0].mesh.size(); ++i) {
        stencils[0].mesh[i] = ((float) i) / sum;
    }

    APRFilter filterfns;
    filterfns.boundary_cond = false;

    timer.start_timer("apr_filter" + std::to_string(stencil_size));
    for (int r = 0; r < num_rep; ++r) {
        ParticleData<float> output;
        filterfns.convolve(apr, stencils, parts, output);
    }
    timer.stop_timer();

    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);
}

template<typename partsType>
inline void bench_apr_convolve_pencil(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size){

    APRTimer timer(true);

    std::vector<PixelData<float>> stencils;
    stencils.resize(1);

    auto it = apr.iterator();

    if(it.number_dimensions() ==3){
        stencils[0].init(stencil_size, stencil_size, stencil_size);
    } else if (it.number_dimensions() ==2){
        stencils[0].init(stencil_size, stencil_size, 1);
    } else if (it.number_dimensions() ==1){
        stencils[0].init(stencil_size, 1, 1);
    }

    // unique stencil elements
    float sum = 0;
    for(int i = 0; i < (int) stencils[0].mesh.size(); ++i) {
        sum += i;
    }
    for(int i = 0; i < (int) stencils[0].mesh.size(); ++i) {
        stencils[0].mesh[i] = ((float) i) / sum;
    }

    APRFilter filterfns;
    filterfns.boundary_cond = false;

    timer.start_timer("apr_filter_pencil" + std::to_string(stencil_size));
    for (int r = 0; r < num_rep; ++r) {
        ParticleData<float> output;
        filterfns.convolve_pencil(apr, stencils, parts, output);
    }
    timer.stop_timer();

    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);
}

template<typename partsType>
inline void bench_pixel_convolve(APR& apr, ParticleData<partsType>& parts, int num_rep,AnalysisData& analysisData,int stencil_size){


    std::vector<PixelData<float>> stencils;
    stencils.resize(1);

    auto it = apr.iterator();

    if(it.number_dimensions() ==3){
        stencils[0].init(stencil_size, stencil_size, stencil_size);
    } else if (it.number_dimensions() ==2){
        stencils[0].init(stencil_size, stencil_size, 1);
    } else if (it.number_dimensions() ==1){
        stencils[0].init(stencil_size, 1, 1);
    }

    // unique stencil elements
    float sum = 0;
    for(int i = 0; i < (int) stencils[0].mesh.size(); ++i) {
        sum += i;
    }
    for(int i = 0; i < (int) stencils[0].mesh.size(); ++i) {
        stencils[0].mesh[i] = ((float) i) / sum;
    }

    const std::vector<int> stencil_shape = {(int) stencils[0].y_num,
                                            (int) stencils[0].x_num,
                                            (int) stencils[0].z_num};

    APRTimer timer(true);

    PixelData<partsType> test_img;
    PixelData<float> test_img_output;

    test_img.init(apr.org_dims(0),apr.org_dims(1),apr.org_dims(2));
    test_img_output.init(test_img);

    timer.start_timer("pixel_filter" + std::to_string(stencil_size));

    const int s_plus = std::floor(stencil_size/2.0);
    const int s_minus = std::ceil(stencil_size/2.0);

    //int z = 0;

    for (int r = 0; r < num_rep; ++r) {
        int z = 0;

        const uint64_t x_num = test_img.x_num;
        const uint64_t y_num = test_img.y_num;
        const uint64_t z_num = test_img.z_num;

#ifdef HAVE_OPENMP
#pragma omp parallel for private(z)
#endif
        for (z = 0; z < (int)  test_img.z_num; ++z) {

            const int offset_max_dim3 = std::min( z + s_plus, (int) (z_num ));
            const int dim3 = std::max(z - s_minus,(int) 0);

            for (int x = 0; x < (int) test_img.x_num; ++x) {

                const int offset_max_dim2 = std::min(x + s_plus, (int) (x_num ));
                const int dim2 = std::max(x - s_minus,(int) 0);

                for (int y = 0; y <  (int) test_img.y_num; ++y) {

                    float temp_int=0;

                    const int dim1 = std::max(y - s_minus,(int) 0);
                    const int offset_max_dim1 = std::min(y + s_plus, (int) (y_num ));

                    int counter = 0;

                    for (int64_t q = dim3; q < offset_max_dim3; ++q) {
                        for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                            const auto off = (k) * y_num + q * y_num*x_num;
                            for (int64_t i = dim1; i < offset_max_dim1; ++i) {

                                temp_int += stencils[0].mesh[counter++]*test_img.mesh[i + off];
                            }
                        }
                    }

                    test_img_output.at(y,x,z) = temp_int;

                }
            }
        }

    }

    timer.stop_timer();

    //Required in all benchmarks
    analysisData.add_timer(timer,test_img.mesh.size(),num_rep);
}


#ifdef APR_USE_CUDA

template<typename partsType>
inline void bench_apr_convolve_cuda(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size){

    APRTimer timer(true);

    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    access.init_gpu();
    tree_access.init_gpu();

    std::vector<float> stencil;
    stencil.resize(stencil_size * stencil_size * stencil_size);

    // unique stencil elements
    float sum = 0;
    for(int i = 0; i < stencil.size(); ++i) {
        sum += i;
    }
    for(int i = 0; i < stencil.size(); ++i) {
        stencil[i] = ((float) i) / sum;
    }

    timings component_times;

    std::string name = "apr_filter_cuda" + std::to_string(stencil_size);
    timer.start_timer(name);

    if(stencil_size == 3) {
        for (int r = 0; r < num_rep; ++r) {
            std::vector<float> output;
            std::vector<float> tree_data;

            timings tmp = isotropic_convolve_333(access, tree_access, parts.data, output, stencil, tree_data);

            component_times.transfer_H2D += tmp.transfer_H2D;
            component_times.run_kernels += tmp.run_kernels;
            component_times.fill_tree += tmp.fill_tree;
            component_times.transfer_D2H += tmp.transfer_D2H;
        }
    } else if(stencil_size == 5) {
        for (int r = 0; r < num_rep; ++r) {
            std::vector<float> output;
            std::vector<float> tree_data;

            timings tmp = isotropic_convolve_555(access, tree_access, parts.data, output, stencil, tree_data);

            component_times.transfer_H2D += tmp.transfer_H2D;
            component_times.run_kernels += tmp.run_kernels;
            component_times.fill_tree += tmp.fill_tree;
            component_times.transfer_D2H += tmp.transfer_D2H;
        }
    } else {
        std::cerr << "APR cuda convolution for stencil_size = " << stencil_size << " is not yet implemented" << std::endl;
    }
    timer.stop_timer();

    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);
    analysisData.add_float_data(name + "_data_transfer_to_device", component_times.transfer_H2D / num_rep);
    analysisData.add_float_data(name + "_run_kernels", component_times.run_kernels / num_rep);
    analysisData.add_float_data(name + "_fill_tree", component_times.fill_tree / num_rep);
    analysisData.add_float_data(name + "_data_transfer_to_host", component_times.transfer_D2H / num_rep);
}


template<typename partsType>
inline void bench_pixel_convolve_cuda(APR& apr,ParticleData<partsType>& parts, int num_rep,AnalysisData& analysisData,int stencil_size) {

    PixelData<float> stencil;

    auto it = apr.iterator();

    if(it.number_dimensions() == 3){
        stencil.init(stencil_size, stencil_size, stencil_size);
    } else if (it.number_dimensions() ==2){
        stencil.init(stencil_size, stencil_size, 1);
    } else if (it.number_dimensions() ==1){
        stencil.init(stencil_size, 1, 1);
    }

    // unique stencil elements
    float sum = 0;
    for(int i = 0; i < stencil.mesh.size(); ++i) {
        sum += i;
    }
    for(int i = 0; i < stencil.mesh.size(); ++i) {
        stencil.mesh[i] = ((float) i) / sum;
    }

    APRTimer timer(true);

    PixelData<partsType> test_img;

    test_img.init(apr.org_dims(0),apr.org_dims(1),apr.org_dims(2));

    timings component_times;

    std::string name = "pixel_filter_cuda" + std::to_string(stencil_size);

    timer.start_timer(name);
    if(stencil_size == 3 && it.number_dimensions() == 3) {
        for (int r = 0; r < num_rep; ++r) {
            PixelData<float> output;
            timings tmp = pixel_convolve_333(test_img, output, stencil);

            component_times.transfer_H2D += tmp.transfer_H2D;
            component_times.run_kernels += tmp.run_kernels;
            component_times.fill_tree += tmp.fill_tree;
            component_times.transfer_D2H += tmp.transfer_D2H;
        }
    } else if(stencil_size == 5 && it.number_dimensions() == 3) {
        for (int r = 0; r < num_rep; ++r) {
            PixelData<float> output;
            timings tmp = pixel_convolve_555(test_img, output, stencil);

            component_times.transfer_H2D += tmp.transfer_H2D;
            component_times.run_kernels += tmp.run_kernels;
            component_times.fill_tree += tmp.fill_tree;
            component_times.transfer_D2H += tmp.transfer_D2H;
        }
    } else {
        std::cerr << "pixel cuda convolution for dim = " << it.number_dimensions() << " and stencil_size = " << stencil_size
            << " is not yet implemented" << std::endl;
    }
    timer.stop_timer();

    analysisData.add_timer(timer,test_img.mesh.size(),num_rep);
    analysisData.add_float_data(name + "_data_transfer_to_device", component_times.transfer_H2D / num_rep);
    analysisData.add_float_data(name + "_run_kernels", component_times.run_kernels / num_rep);
    analysisData.add_float_data(name + "_fill_tree", component_times.fill_tree / num_rep);
    analysisData.add_float_data(name + "_data_transfer_to_host", component_times.transfer_D2H / num_rep);
}

#endif //APR_USE_CUDA

#endif //LIBAPR_FILTERBENCHMARKS_HPP
