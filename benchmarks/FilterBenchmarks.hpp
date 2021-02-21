//
// Created by Joel Jonsson on 2019-07-18.
//

#ifndef LIBAPR_FILTERBENCHMARKS_HPP
#define LIBAPR_FILTERBENCHMARKS_HPP

#include <algorithm>
#include <iostream>

#include "APRBenchHelper.hpp"
#include "numerics/APRFilter.hpp"
#include "numerics/APRStencil.hpp"

#ifdef APR_USE_CUDA
#include "numerics/APRNumericsGPU.hpp"
#include "numerics/PixelNumericsGPU.hpp"
#endif


template<typename partsType>
inline void bench_apr_convolve(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size, bool ds_stencil=true){

    APRTimer timer(true);

    Stencil<float> stenc;

    auto it = apr.iterator();

    if(it.number_dimensions() ==3){
        stenc.init(stencil_size, stencil_size, stencil_size);
    } else if (it.number_dimensions() ==2){
        stenc.init(stencil_size, stencil_size, 1);
    } else if (it.number_dimensions() ==1){
        stenc.init(stencil_size, 1, 1);
    }

    // unique stencil elements
    float sz = stenc.size();
    float sum = sz * (sz - 1) / 2;
    for(int i = 0; i < (int) stenc.size(); ++i) {
        stenc[i] = ((float) i) / sum;
    }

    int nstencils = ds_stencil ? it.level_max() - it.level_min() : 1;
    MultiStencil<float> stencils(stenc, nstencils);
    stencils.restrict_stencils(nstencils, false);

    ParticleData<float> output;
    int burn_in = std::max( std::min(num_rep / 10, 10), 1 );
    for (int r = 0; r < burn_in; ++r) {
        APRFilter::convolve(apr, stencils, parts, output, false);
    }

    timer.start_timer("apr_filter" + std::to_string(stencil_size));
    for (int r = 0; r < num_rep; ++r) {
        APRFilter::convolve(apr, stencils, parts, output, false);
    }
    timer.stop_timer();

    analysisData.add_timer(timer,apr.total_number_particles(), (float)num_rep);
}

template<typename partsType>
inline void bench_apr_convolve_pencil(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size, bool ds_stencil=true){

    APRTimer timer(true);

    Stencil<float> stenc;

    auto it = apr.iterator();

    if(it.number_dimensions() ==3){
        stenc.init(stencil_size, stencil_size, stencil_size);
    } else if (it.number_dimensions() ==2){
        stenc.init(stencil_size, stencil_size, 1);
    } else if (it.number_dimensions() ==1){
        stenc.init(stencil_size, 1, 1);
    }

    // unique stencil elements
    float sz = stenc.size();
    float sum = sz * (sz - 1) / 2;
    for(int i = 0; i < (int) stenc.size(); ++i) {
        stenc[i] = ((float) i) / sum;
    }

    int nstencils = ds_stencil ? it.level_max() - it.level_min() : 1;
    MultiStencil<float> stencils(stenc, nstencils);
    stencils.restrict_stencils(nstencils, false);

    ParticleData<float> output;
    int burn_in = std::max( std::min(num_rep / 10, 10), 1 );
    for (int r = 0; r < burn_in; ++r) {
        APRFilter::convolve_pencil(apr, stencils, parts, output, false);
    }

    timer.start_timer("apr_filter_pencil" + std::to_string(stencil_size));
    for (int r = 0; r < num_rep; ++r) {
        APRFilter::convolve_pencil(apr, stencils, parts, output, false);
    }
    timer.stop_timer();

    analysisData.add_timer(timer,apr.total_number_particles(), (float)num_rep);
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
#pragma omp parallel for default(shared) private(z)
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


template<typename partsType>
inline void check_cpu_times(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData){

    APRTimer timer(true);

    auto apr_iterator = apr.iterator();
    for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {

        //ind[level].resize( apr_iterator.z_num(level)*apr_iterator.x_num(level),0);

    }
    timer.start_timer("cpu_it");

    for (int r = 0; r < num_rep; ++r) {

        std::vector<std::vector<int>> ind;

        ind.resize(apr_iterator.level_max()+1);

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

            for (z = 0; z < apr_iterator.z_num(level-1); z++) {
                for (x = 0; x < apr_iterator.x_num(level-1); ++x) {
                    auto begin = apr_iterator.begin(level, 2*z, 2*x);
                    auto end = apr_iterator.end();
                    if(begin != end) {
                        ind[level].push_back(2*x + 2*z * apr_iterator.x_num(level));
                    }
                }
            }
        }
    }

    timer.stop_timer();
    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);
}



#ifdef APR_USE_CUDA


template<typename partsType>
inline void bench_initial_steps_cuda(APR& apr, ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_apr") {

    APRTimer timer(false);

    auto apr_it = apr.iterator();

    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    analysisData.add_float_data(name + "_y_vec_send",apr_it.total_number_particles(apr_it.level_max()-1));
    analysisData.add_float_data(name + "_apr_xz_size",access.linearAccess->xz_end_vec.size());
    analysisData.add_float_data(name + "_tree_xz_size",tree_access.linearAccess->xz_end_vec.size());

    /// access GPU initialization

    for(int i = 0; i < num_rep/10; ++i) {
        tree_access.init_gpu();
        access.init_gpu(tree_access);
    }

    cudaDeviceSynchronize();

    timer.start_timer("init_access");
    for(int i = 0; i < num_rep; ++i) {
        tree_access.init_gpu();
        access.init_gpu(tree_access);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_init_access", timer.timings.back() / num_rep);

    /// compute ne rows tree

    VectorData<int> ne_counter_ds;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu;

    for(int i = 0; i < num_rep/10; ++i) {
        compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
    }
    cudaDeviceSynchronize();

    timer.start_timer("compute_ne_rows_tree");
    for(int i = 0; i < num_rep; ++i) {
        compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_compute_ne_rows_tree", timer.timings.back() / num_rep);

    /// compute ne rows 333

    VectorData<int> ne_counter_333;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_333_gpu;

    for(int i = 0; i < num_rep/10; ++i) {
        compute_ne_rows_cuda<16, 32>(access, ne_counter_333, ne_rows_333_gpu, 2);
    }
    cudaDeviceSynchronize();

    timer.start_timer("compute_ne_rows_333");
    for(int i = 0; i < num_rep; ++i) {
        compute_ne_rows_cuda<16, 32>(access, ne_counter_333, ne_rows_333_gpu, 2);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_compute_ne_rows_333", timer.timings.back() / num_rep);


    /// compute ne rows 555

    VectorData<int> ne_counter_555;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_555_gpu;

    for(int i = 0; i < num_rep/10; ++i) {
        compute_ne_rows_cuda<16, 32>(access, ne_counter_555, ne_rows_555_gpu, 4);
    }
    cudaDeviceSynchronize();

    timer.start_timer("compute_ne_rows_555");
    for(int i = 0; i < num_rep; ++i) {
        compute_ne_rows_cuda<16, 32>(access, ne_counter_555, ne_rows_555_gpu, 4);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_compute_ne_rows_555", timer.timings.back() / num_rep);


    analysisData.add_float_data(name + "_ne_rows_tree_size", ne_counter_ds.back());
    analysisData.add_float_data(name + "_ne_rows_333_size", ne_counter_333.back());
    analysisData.add_float_data(name + "_ne_rows_555_size", ne_counter_555.back());

    /// send data to GPU
    ScopedCudaMemHandler<partsType*, JUST_ALLOC> input_gpu(parts.data.data(), parts.data.size());
    for(int i = 0; i < num_rep/10; ++i) {
        input_gpu.copyH2D();
    }
    cudaDeviceSynchronize();

    timer.start_timer("send particles");
    for(int i = 0; i < num_rep; ++i) {
        input_gpu.copyH2D();
    }
    cudaDeviceSynchronize();
    timer.stop_timer();
    analysisData.add_float_data(name + "_send_particles_time", timer.timings.back() / num_rep);


    /// fill tree

    VectorData<float> tree_data;
    tree_data.resize(tree_access.total_number_particles());
    ScopedCudaMemHandler<float*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());

    cudaDeviceSynchronize();

    for(int i = 0; i < num_rep/10; ++i) {
        downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    }

    cudaDeviceSynchronize();

    timer.start_timer("fill_tree");
    for(int i = 0; i < num_rep; ++i) {
        downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_fill_tree", timer.timings.back() / num_rep);

    /// without ne rows

    for(int i = 0; i < num_rep/10; ++i) {
        downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    }

    cudaDeviceSynchronize();

    timer.start_timer("fill_tree_alt");
    for(int i = 0; i < num_rep; ++i) {
        downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_fill_tree_alt", timer.timings.back() / num_rep);


    /// Copy data to host
    for(int i = 0; i < num_rep/10; ++i) {
        input_gpu.copyD2H();
    }
    cudaDeviceSynchronize();

    timer.start_timer("retrieve particles");
    for(int i = 0; i < num_rep; ++i) {
        input_gpu.copyD2H();
    }
    cudaDeviceSynchronize();
    timer.stop_timer();
    analysisData.add_float_data(name + "_retrieve_particles_time", timer.timings.back() / num_rep);
}


template<typename partsType>
inline void bench_apr_convolve_cuda_333(APR& apr, ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_apr",
                                        bool include_reflective = true, bool include_alt = true) {

    APRTimer timer(false);

    VectorData<float> stencil;
    stencil.resize(27);

    // unique stencil elements
    float sum = 13.0 * 27;
    for(size_t i = 0; i < stencil.size(); ++i) {
        stencil[i] = ((float) i) / sum;
    }

    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    VectorData<float> tree_data, output;

    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());

    /// compute nonempty rows
    VectorData<int> ne_counter_ds;
    VectorData<int> ne_counter;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;

    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
    compute_ne_rows_cuda<16, 32>(access, ne_counter, ne_rows_gpu, 2);

    /// allocate GPU memory
    ScopedCudaMemHandler<partsType*, JUST_ALLOC> input_gpu(parts.data.data(), parts.data.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    cudaDeviceSynchronize();


    /// conv 333

    for(int i = 0; i < num_rep/10; ++i) {
        isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, false);
    }
    cudaDeviceSynchronize();

    timer.start_timer("conv_333");
    for(int i = 0; i < num_rep; ++i) {
        isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, false);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_conv_333", timer.timings.back() / num_rep);


    /// conv 333 reflective

    if(include_reflective) {
        for (int i = 0; i < num_rep/10; ++i) {
            isotropic_convolve_333_reflective(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                                              tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, false);
        }
        cudaDeviceSynchronize();

        timer.start_timer("conv_333_reflective");
        for (int i = 0; i < num_rep; ++i) {
            isotropic_convolve_333_reflective(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                                              tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, false);
        }
        cudaDeviceSynchronize();
        timer.stop_timer();

        analysisData.add_float_data(name + "_conv_333_reflective", timer.timings.back() / num_rep);
    }

    /// conv 333 alt

    if(include_alt) {
        for (int i = 0; i < num_rep/10; ++i) {
            isotropic_convolve_333_alt(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), false);
        }
        cudaDeviceSynchronize();

        timer.start_timer("conv_333_alt");
        for (int i = 0; i < num_rep; ++i) {
            isotropic_convolve_333_alt(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), false);
        }
        cudaDeviceSynchronize();
        timer.stop_timer();

        analysisData.add_float_data(name + "_conv_333_alt", timer.timings.back() / num_rep);
    }
}


template<typename partsType>
inline void bench_apr_convolve_cuda_555(APR& apr, ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_apr",
                                        bool include_reflective = true, bool include_alt = true, bool include_ds_stencil = true) {

    APRTimer timer(false);

    VectorData<float> stencil;
    stencil.resize(125);

    // unique stencil elements
    float sum = 62.0 * 125;
    for(size_t i = 0; i < stencil.size(); ++i) {
        stencil[i] = ((float) i) / sum;
    }

    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    VectorData<float> tree_data, output;

    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());

    /// compute nonempty rows
    VectorData<int> ne_counter_ds;
    VectorData<int> ne_counter;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_ds_gpu;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_gpu;

    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
    compute_ne_rows_cuda<16, 32>(access, ne_counter, ne_rows_gpu, 4);

    /// allocate GPU memory
    ScopedCudaMemHandler<partsType*, JUST_ALLOC> input_gpu(parts.data.data(), parts.data.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
    cudaDeviceSynchronize();


    /// conv 555

    for(int i = 0; i < num_rep/10; ++i) {
        isotropic_convolve_555(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);
    }
    cudaDeviceSynchronize();

    timer.start_timer("conv_555");
    for(int i = 0; i < num_rep; ++i) {
        isotropic_convolve_555(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_conv_555", timer.timings.back() / num_rep);


    /// conv 555 reflective

    if(include_reflective) {
        for (int i = 0; i < num_rep/10; ++i) {
            isotropic_convolve_555_reflective(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                                              tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);
        }
        cudaDeviceSynchronize();

        timer.start_timer("conv_555_reflective");
        for (int i = 0; i < num_rep; ++i) {
            isotropic_convolve_555_reflective(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                                              tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);
        }
        cudaDeviceSynchronize();
        timer.stop_timer();

        analysisData.add_float_data(name + "_conv_555_reflective", timer.timings.back() / num_rep);
    }

    /// conv 555 alt

    if(include_alt) {
        for (int i = 0; i < num_rep/10; ++i) {
            isotropic_convolve_555_alt(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
        }
        cudaDeviceSynchronize();

        timer.start_timer("conv_555_alt");
        for (int i = 0; i < num_rep; ++i) {
            isotropic_convolve_555_alt(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
        }
        cudaDeviceSynchronize();
        timer.stop_timer();

        analysisData.add_float_data(name + "_conv_555_alt", timer.timings.back() / num_rep);
    }

    /// conv 555 downsample stencil

    if(include_ds_stencil) {
        VectorData<int> ne_counter_333;
        ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_333_gpu;

        compute_ne_rows_cuda<16, 32>(access, ne_counter_333, ne_rows_333_gpu, 2);
        cudaDeviceSynchronize();

        for (int i = 0; i < num_rep/10; ++i) {
            isotropic_convolve_555_ds(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(),
                                      ne_rows_gpu.get(), ne_counter, ne_rows_333_gpu.get(), ne_counter_333);
        }
        cudaDeviceSynchronize();

        timer.start_timer("conv_555_alt");
        for (int i = 0; i < num_rep; ++i) {
            isotropic_convolve_555_ds(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(),
                                      ne_rows_gpu.get(), ne_counter, ne_rows_333_gpu.get(), ne_counter_333);
        }
        cudaDeviceSynchronize();
        timer.stop_timer();

        analysisData.add_float_data(name + "_conv_555_ds", timer.timings.back() / num_rep);
    }
}


template<typename partsType>
inline void bench_apr_convolve_cuda_555_full(APR& apr, ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_apr") {

    APRTimer timer(false);

    VectorData<float> stencil;
    stencil.resize(125);

    // unique stencil elements
    float sum = 62.0 * 125;
    for (size_t i = 0; i < stencil.size(); ++i) {
        stencil[i] = ((float) i) / sum;
    }

    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    VectorData<float> tree_data, output;

    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());

    /// allocate GPU memory
    ScopedCudaMemHandler<partsType *, JUST_ALLOC> input_gpu(parts.data.data(), parts.data.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    /// copy data to device
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    for (int i = 0; i < num_rep / 10; ++i) {
        /// compute nonempty rows
        VectorData<int> ne_counter_ds;
        VectorData<int> ne_counter;
        ScopedCudaMemHandler<int *, JUST_ALLOC> ne_rows_ds_gpu;
        ScopedCudaMemHandler<int *, JUST_ALLOC> ne_rows_gpu;

        compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
        compute_ne_rows_cuda<16, 32>(access, ne_counter, ne_rows_gpu, 4);

        /// fill tree
        downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
        cudaDeviceSynchronize();

        /// convolve
        isotropic_convolve_555(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                               tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);
    }
    cudaDeviceSynchronize();

    timer.start_timer("conv_555_with_precomp");
    for (int i = 0; i < num_rep; ++i) {
        /// compute nonempty rows
        VectorData<int> ne_counter_ds;
        VectorData<int> ne_counter;
        ScopedCudaMemHandler<int *, JUST_ALLOC> ne_rows_ds_gpu;
        ScopedCudaMemHandler<int *, JUST_ALLOC> ne_rows_gpu;

        compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
        compute_ne_rows_cuda<16, 32>(access, ne_counter, ne_rows_gpu, 4);

        /// fill tree
        downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
        cudaDeviceSynchronize();

        /// convolve
        isotropic_convolve_555(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                               tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_conv_555_full", timer.timings.back() / num_rep);
}


template<typename partsType>
inline void bench_apr_convolve_cuda_333_full(APR& apr, ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_apr") {

    APRTimer timer(false);

    VectorData<float> stencil;
    stencil.resize(27);

    // unique stencil elements
    float sum = 13.0 * 27;
    for (size_t i = 0; i < stencil.size(); ++i) {
        stencil[i] = ((float) i) / sum;
    }

    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    VectorData<float> tree_data, output;

    tree_data.resize(tree_access.total_number_particles());
    output.resize(access.total_number_particles());

    /// allocate GPU memory
    ScopedCudaMemHandler<partsType *, JUST_ALLOC> input_gpu(parts.data.data(), parts.data.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> tree_data_gpu(tree_data.data(), tree_data.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    /// copy data to device
    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    for (int i = 0; i < num_rep / 10; ++i) {
        /// compute nonempty rows
        VectorData<int> ne_counter_ds;
        VectorData<int> ne_counter;
        ScopedCudaMemHandler<int *, JUST_ALLOC> ne_rows_ds_gpu;
        ScopedCudaMemHandler<int *, JUST_ALLOC> ne_rows_gpu;

        compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
        compute_ne_rows_cuda<16, 32>(access, ne_counter, ne_rows_gpu, 2);

        /// fill tree
        downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
        cudaDeviceSynchronize();

        /// convolve
        isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                               tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, false);
    }
    cudaDeviceSynchronize();

    timer.start_timer("conv_333_with_precomp");
    for (int i = 0; i < num_rep; ++i) {
        /// compute nonempty rows
        VectorData<int> ne_counter_ds;
        VectorData<int> ne_counter;
        ScopedCudaMemHandler<int *, JUST_ALLOC> ne_rows_ds_gpu;
        ScopedCudaMemHandler<int *, JUST_ALLOC> ne_rows_gpu;

        compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_counter_ds, ne_rows_ds_gpu);
        compute_ne_rows_cuda<16, 32>(access, ne_counter, ne_rows_gpu, 2);

        /// fill tree
        downsample_avg(access, tree_access, input_gpu.get(), tree_data_gpu.get(), ne_rows_ds_gpu.get(), ne_counter_ds);
        cudaDeviceSynchronize();

        /// convolve
        isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(),
                               tree_data_gpu.get(), ne_rows_gpu.get(), ne_counter, false);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_conv_333_full", timer.timings.back() / num_rep);
}




template<typename partsType>
inline void bench_richardson_lucy_apr(APR& apr, ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_apr", int niter = 10) {

    APRTimer timer(false);

    /// initialize gpu access
    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    /// initialize and downsample stencil
    VectorData<float> stencil;
    stencil.resize(125, 1.0f/125.0f);
    VectorData<float> stencil_vec;
    APRStencil::get_downsampled_stencils(stencil, stencil_vec, apr.level_max()-apr.level_min(), true);

    VectorData<float> output;
    output.resize(access.total_number_particles());

    /// allocate GPU memory
    ScopedCudaMemHandler<partsType *, JUST_ALLOC> input_gpu(parts.data.data(), parts.data.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> output_gpu(output.data(), output.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> stencil_gpu(stencil_vec.data(), stencil_vec.size());

    input_gpu.copyH2D();

    // burn-in
    for(int i = 0; i < num_rep/10; ++i) {
        richardson_lucy(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), stencil_gpu.get(), 5, niter, true);
    }
    cudaDeviceSynchronize();

    timer.start_timer("richardson lucy");
    for(int i = 0; i < num_rep; ++i) {
        richardson_lucy(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), stencil_gpu.get(), 5, niter, true);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_richardson_lucy_555_ds_" + std::to_string(niter), timer.timings.back() / num_rep);
}


template<typename partsType>
inline void bench_pixel_convolve_cuda_333(APR& apr,ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_pixel", bool reflective = true) {

    APRTimer timer(false);

    PixelData<partsType> input(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), 3);
    PixelData<float> output(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2));

    VectorData<float> stencil;
    stencil.resize(27);
    // unique stencil elements
    float sum = 13.0f * 27;
    for(size_t i = 0; i < stencil.size(); ++i) {
        stencil[i] = ((float) i) / sum;
    }

    ScopedCudaMemHandler<partsType*, JUST_ALLOC> input_gpu(input.mesh.get(), input.mesh.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> output_gpu(output.mesh.get(), output.mesh.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    /// conv 333

    for(int i = 0; i < num_rep/10; ++i) {
        convolve_pixel_333(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
    }
    cudaDeviceSynchronize();

    timer.start_timer("conv_333");
    for(int i = 0; i < num_rep; ++i) {
        convolve_pixel_333(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_conv_333", timer.timings.back() / num_rep);


    /// conv 333 reflective

    if(reflective) {
        for (int i = 0; i < num_rep/10; ++i) {
            convolve_pixel_333_reflective(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
        }
        cudaDeviceSynchronize();

        timer.start_timer("conv_333_reflective");
        for (int i = 0; i < num_rep; ++i) {
            convolve_pixel_333_reflective(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
        }
        cudaDeviceSynchronize();
        timer.stop_timer();

        analysisData.add_float_data(name + "_conv_333_reflective", timer.timings.back() / num_rep);
    }
}


template<typename partsType>
inline void bench_pixel_convolve_cuda_555(APR& apr,ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_pixel", bool reflective = true) {

    APRTimer timer(false);

    PixelData<partsType> input(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), 3);
    PixelData<float> output(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2));

    VectorData<float> stencil;
    stencil.resize(125);
    // unique stencil elements
    float sum = 62.0f * 125;
    for(size_t i = 0; i < stencil.size(); ++i) {
        stencil[i] = ((float) i) / sum;
    }

    ScopedCudaMemHandler<partsType*, JUST_ALLOC> input_gpu(input.mesh.get(), input.mesh.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> output_gpu(output.mesh.get(), output.mesh.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    /// conv 555

    for(int i = 0; i < num_rep/10; ++i) {
        convolve_pixel_555(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
    }
    cudaDeviceSynchronize();

    timer.start_timer("conv_555");
    for(int i = 0; i < num_rep; ++i) {
        convolve_pixel_555(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_conv_555", timer.timings.back() / num_rep);


    /// conv 555 reflective

    if(reflective) {
        for (int i = 0; i < num_rep/10; ++i) {
            convolve_pixel_555_reflective(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
        }
        cudaDeviceSynchronize();

        timer.start_timer("conv_555_reflective");
        for (int i = 0; i < num_rep; ++i) {
            convolve_pixel_555_reflective(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), input.y_num, input.x_num, input.z_num);
        }
        cudaDeviceSynchronize();
        timer.stop_timer();

        analysisData.add_float_data(name + "_conv_555_reflective", timer.timings.back() / num_rep);
    }
}





template<typename partsType>
inline void bench_richardson_lucy_pixel(APR& apr,ParticleData<partsType>& parts, int num_rep, AnalysisData& analysisData, std::string name = "bench_pixel", int niter = 10) {

    APRTimer timer(false);

    PixelData<partsType> input(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), 3);
    PixelData<float> output(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2));

    std::vector<int> dims = {static_cast<int>(input.z_num), static_cast<int>(input.x_num), static_cast<int>(input.y_num)};

    VectorData<float> stencil;
    stencil.resize(125, 1.0f/125.0f);

    ScopedCudaMemHandler<partsType *, JUST_ALLOC> input_gpu(input.mesh.get(), input.mesh.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> output_gpu(output.mesh.get(), output.mesh.size());
    ScopedCudaMemHandler<float *, JUST_ALLOC> stencil_gpu(stencil.data(), stencil.size());

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    cudaDeviceSynchronize();

    for(int i = 0; i < num_rep/10; ++i) {
        richardson_lucy_pixel(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), stencil_gpu.get(), 5, input.mesh.size(), niter, dims);
    }
    cudaDeviceSynchronize();

    timer.start_timer("richardson lucy");
    for(int i = 0; i < num_rep; ++i) {
        richardson_lucy_pixel(input_gpu.get(), output_gpu.get(), stencil_gpu.get(), stencil_gpu.get(), 5, input.mesh.size(), niter, dims);
    }
    cudaDeviceSynchronize();
    timer.stop_timer();

    analysisData.add_float_data(name + "_richardson_lucy_555_" + std::to_string(niter), timer.timings.back() / num_rep);
}

#endif //APR_USE_CUDA

#endif //LIBAPR_FILTERBENCHMARKS_HPP
