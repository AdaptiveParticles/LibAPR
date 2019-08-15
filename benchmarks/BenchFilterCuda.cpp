//
// Created by Joel Jonsson on 2019-07-18.
//
//
// Created by cheesema on 2019-07-03.
//
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2019
const char* usage = R"(
Benchmarking iteration performance.

Usage:

BenchIteration

)";

#include "FilterBenchmarks.hpp"

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

int main(int argc, char **argv) {

    // INPUT PARSING
#ifndef APR_USE_CUDA
    std::cerr << "CUDA is disabled" << std::endl;
    //return 1;
#endif

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
#ifdef APR_USE_CUDA
        bench_apr_convolve_cuda(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, 3);
        //bench_apr_convolve_cuda(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, 5);
        //bench_check_blocks(apr, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData);
        
        //bench_333_old(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, 3);
        //bench_333_new(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, 3);
#endif

        check_cpu_times(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData);

        if(i==0){
            /*
            * Pixel benchmarks (These are content independent)
            */

#ifdef APR_USE_CUDA
            bench_pixel_convolve_cuda(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,3);
            //bench_pixel_convolve_cuda_basic(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,3);
            //bench_pixel_convolve_cuda(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,5);
#endif

        }
    }


    benchAPRHelper.analysisData.init(analysis_file_name,"filter_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}
