
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2019
const char* usage = R"(
Benchmarking iteration performance.

Usage:

BenchIteration

)";


#include <algorithm>
#include <iostream>

#include "BenchAPRHelper.hpp"

int main(int argc, char **argv) {

    // INPUT PARSING

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

        bench_apr_iteration_old(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);

        //put benchmark funtions here..
        bench_apr_iteration(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);

        if(i==0){
            /*
            * Pixel benchmarks (These are content independent)
            */
            bench_pixel_iteration(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);
        }

    }


    benchAPRHelper.analysisData.init(analysis_file_name,"iteration_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}
