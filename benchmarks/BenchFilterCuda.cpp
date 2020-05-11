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
//    return 1;
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

        /// data type can be changed here
        ParticleData<uint16_t> parts;
        APR apr;

        benchAPRHelper.generate_dataset(i,apr,parts);

        ParticleData<float> floatparts(apr.total_number_particles());
        for(int j = 0; j < parts.size(); ++j) {
            floatparts[j] = parts[j];
        }

        ///put benchmark funtions here..
#ifdef APR_USE_CUDA
        bench_initial_steps_cuda(apr, floatparts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, "apr");

        bench_apr_convolve_cuda_333(apr, floatparts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, "apr", true, true);
        bench_apr_convolve_cuda_555(apr, floatparts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, "apr", true, true, true);

        bench_apr_convolve_cuda_333_full(apr, floatparts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, "apr");
        bench_apr_convolve_cuda_555_full(apr, floatparts, benchAPRHelper.get_number_reps(), benchAPRHelper.analysisData, "apr");

        if(options.bench_lr) {
            bench_richardson_lucy_apr(apr, floatparts, benchAPRHelper.get_number_reps()/10, benchAPRHelper.analysisData, "apr", 10);
            bench_richardson_lucy_apr(apr, floatparts, benchAPRHelper.get_number_reps()/10, benchAPRHelper.analysisData, "apr", 30);
            bench_richardson_lucy_apr(apr, floatparts, benchAPRHelper.get_number_reps()/10, benchAPRHelper.analysisData, "apr", 50);
        }

        ///pixel convolution is content-independent
        if(i == 0 && !options.no_pixel) {
            bench_pixel_convolve_cuda_333(apr, floatparts, benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData, "pixel", true);
            bench_pixel_convolve_cuda_555(apr, floatparts, benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData, "pixel", true);

            if(options.bench_lr) {
                bench_richardson_lucy_pixel(apr, floatparts, benchAPRHelper.get_number_reps()/10, benchAPRHelper.analysisData, "pixel", 10);
                bench_richardson_lucy_pixel(apr, floatparts, benchAPRHelper.get_number_reps()/10, benchAPRHelper.analysisData, "pixel", 30);
                bench_richardson_lucy_pixel(apr, floatparts, benchAPRHelper.get_number_reps()/10, benchAPRHelper.analysisData, "pixel", 50);
            }
        }
#endif

    }

    benchAPRHelper.analysisData.init(analysis_file_name,"filter_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}
