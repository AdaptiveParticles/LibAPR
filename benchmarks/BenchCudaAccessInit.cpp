//
// Created by Joel Jonsson on 2019-08-05.
//

#include <algorithm>
#include <iostream>

#include "APRBenchHelper.hpp"

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

template<typename partsType>
inline void bench_access_full(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);

template<typename partsType>
inline void bench_access_partial(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);


int main(int argc, char **argv) {

    // INPUT PARSING

    cmdLineBenchOptions options = read_bench_command_line_options(argc, argv);

    // Filename
    std::string analysis_file_name = options.output_dir + options.analysis_file_name;

    APRBenchHelper benchAPRHelper;
    benchAPRHelper.initialize_benchmark(options);

    /*
     * APR benchmarks (Results depend on the content)
     */
    for (int i = 0; i < benchAPRHelper.number_datsets(); ++i) {

        ParticleData<uint16_t> parts;
        APR apr;

        benchAPRHelper.generate_dataset(i,apr,parts);

        //put benchmark funtions here..
        bench_access_full(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);
        bench_access_partial(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);
    }

    benchAPRHelper.analysisData.init(analysis_file_name,"access_init_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();
}


template<typename partsType>
inline void bench_access_full(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData) {

    APRTimer timer(false);
    APRTimer timer2(false);

    float apr_time = 0;
    float tree_time = 0;

    for(int r = 0; r < 5; ++r) {
        auto access = apr.gpuAPRHelper();
        access.init_gpu();

        auto tree_access = apr.gpuTreeHelper();
        tree_access.init_gpu();
    }

    timer.start_timer("init_full");
    for(int r = 0; r < num_rep; ++r) {
        timer2.start_timer("apr access");
        auto access = apr.gpuAPRHelper();
        access.init_gpu();
        error_check ( cudaDeviceSynchronize() )
        error_check( cudaPeekAtLastError() )
        timer2.stop_timer();
        apr_time += timer2.timings.back();

        timer2.start_timer("tree access");
        auto tree_access = apr.gpuTreeHelper();
        tree_access.init_gpu();
        error_check ( cudaDeviceSynchronize() )
        error_check( cudaPeekAtLastError() )
        timer2.stop_timer();
        tree_time += timer2.timings.back();
    }
    timer.stop_timer();

    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);
    analysisData.add_float_data("init_full_apr", apr_time / num_rep);
    analysisData.add_float_data("init_full_tree", tree_time / num_rep);
}


template<typename partsType>
inline void bench_access_partial(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData) {

    APRTimer timer(false);
    APRTimer timer2(false);

    float apr_time = 0;
    float tree_time = 0;

    for(int r = 0; r < 5; ++r) {
        auto it = apr.iterator();

        auto tree_access = apr.gpuTreeHelper();
        tree_access.init_gpu();

        auto access = apr.gpuAPRHelper();
        access.init_gpu(it.total_number_particles(it.level_max()-1), tree_access);

    }

    timer.start_timer("init_partial");
    for(int r = 0; r < num_rep; ++r) {

        timer2.start_timer("tree access");
        auto tree_access = apr.gpuTreeHelper();
        tree_access.init_gpu();
        error_check ( cudaDeviceSynchronize() )
        error_check( cudaPeekAtLastError() )
        timer2.stop_timer();
        tree_time += timer2.timings.back();

        timer2.start_timer("apr access");
        auto it = apr.iterator();
        auto access = apr.gpuAPRHelper();
        access.init_gpu(it.total_number_particles(it.level_max()-1), tree_access);
        error_check ( cudaDeviceSynchronize() )
        error_check( cudaPeekAtLastError() )
        timer2.stop_timer();
        apr_time += timer2.timings.back();
    }
    timer.stop_timer();

    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);
    analysisData.add_float_data("init_partial_apr", apr_time / num_rep);
    analysisData.add_float_data("init_partial_tree", tree_time / num_rep);
}