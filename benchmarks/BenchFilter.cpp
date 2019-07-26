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

        //bench_apr_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,1);
        bench_apr_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,3);
        bench_apr_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,5);

        //bench_apr_convolve_pencil(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,1);
        bench_apr_convolve_pencil(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,3);
        bench_apr_convolve_pencil(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,5);

        if(i==0){
            /*
            * Pixel benchmarks (These are content independent)
            */

            //bench_pixel_convolve(apr,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,1);
            bench_pixel_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,3);
            bench_pixel_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,5);

        }
    }


    benchAPRHelper.analysisData.init(analysis_file_name,"filter_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}