//
// Created by Joel Jonsson on 2019-07-26.
//

#ifdef APR_USE_CUDA
#include "../test/GPUAPR.hpp"
#endif

#include "APRBenchHelper.hpp"

int main(int argc, char **argv) {

    // INPUT PARSING
#ifndef APR_USE_CUDA
    std::cerr << "CUDA is disabled" << std::endl;
    return 1;
#endif

    cmdLineBenchOptions options = read_bench_command_line_options(argc, argv);

    // Filename
    std::string analysis_file_name = options.output_dir + options.analysis_file_name;

    APRBenchHelper benchAPRHelper;
    benchAPRHelper.initialize_benchmark(options);

    int warmup = 5;

#ifdef APR_USE_CUDA
    for (int i = 0; i < benchAPRHelper.number_datsets(); ++i) {

        ParticleData<uint16_t> parts;
        APR apr;

        benchAPRHelper.generate_dataset(i, apr, parts);

        auto access = apr.gpuAPRHelper();
        access.init_gpu();

        bench_sequential_iterate<1>(access, warmup); /// warmup

        double ts = bench_sequential_iterate<1>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("sequential_iteration_1", ts);

        ts = bench_sequential_iterate<2>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("sequential_iteration_2", ts);

        ts = bench_sequential_iterate<4>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("sequential_iteration_4", ts);

        ts = bench_sequential_iterate<8>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("sequential_iteration_8", ts);

        ts = bench_sequential_iterate<16>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("sequential_iteration_16", ts);


        bench_chunked_iterate<1, 128>(access, warmup); /// warmup

        double tc = bench_chunked_iterate<1, 128>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("chunked_iteration_128_1", tc);

        tc = bench_chunked_iterate<2, 128>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("chunked_iteration_128_2", tc);

        tc = bench_chunked_iterate<4, 128>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("chunked_iteration_128_4", tc);

        tc = bench_chunked_iterate<8, 128>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("chunked_iteration_128_8", tc);

        tc = bench_chunked_iterate<1, 256>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("chunked_iteration_256_1", tc);

        tc = bench_chunked_iterate<2, 256>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("chunked_iteration_256_2", tc);

        tc = bench_chunked_iterate<4, 256>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("chunked_iteration_256_4", tc);

        tc = bench_chunked_iterate<8, 256>(access, benchAPRHelper.get_number_reps());
        benchAPRHelper.analysisData.add_float_data("chunked_iteration_256_8", tc);

    }
    benchAPRHelper.analysisData.init(analysis_file_name,"iteration_benchmark",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();
#endif

    return 0;
}
