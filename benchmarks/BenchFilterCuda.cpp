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

int main(int argc, char **argv) {

    // INPUT PARSING
#ifndef APR_USE_CUDA
    std::cerr << "CUDA is disabled" << std::endl;
    return 1;
#endif

    cmdLineBenchOptions options = read_bench_command_line_options(argc, argv);

    // Filename
    std::string analysis_file_name = options.output_dir + options.analysis_file_name;

    BenchAPRHelper benchAPRHelper;
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
        //bench_apr_convolve_cuda(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, 3);
        //bench_apr_convolve_cuda(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, 5);
        //bench_check_blocks(apr, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData);
        
        bench_333_old(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, 3);
        bench_333_new(apr, parts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, 3);
#endif

        if(i==0){
            /*
            * Pixel benchmarks (These are content independent)
            */

#ifdef APR_USE_CUDA
            //bench_pixel_convolve_cuda(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,3);
            //bench_pixel_convolve_cuda(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,5);
#endif

        }
    }


    benchAPRHelper.analysisData.init(analysis_file_name,"filter_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}
