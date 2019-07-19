
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2019
const char* usage = R"(
Benchmarking iteration performance.

Usage:

BenchPipeline

)";


#include <algorithm>
#include <iostream>

#include "APRBenchHelper.hpp"

template<typename partsType>
void bench_apr_pipeline(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);

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

        std::cout << "generated" << std::endl;

        //put benchmark funtions here..
        bench_apr_pipeline(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);

    }


    benchAPRHelper.analysisData.init(analysis_file_name,"iteration_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}
template<typename partsType>
inline void bench_apr_pipeline(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    APRTimer timer(true);

    PixelData<partsType> pipelineImage;

    APRReconstruction::interp_img(apr,pipelineImage,parts);

    APRConverter<partsType> aprConverter;

    aprConverter.par.auto_parameters = false;
    aprConverter.par.grad_th = 10;

    std::vector<APR> apr_vec;
    apr_vec.resize(num_rep);

    std::vector<ParticleData<partsType>> part_vec;
    part_vec.resize(num_rep);

    timer.start_timer("pipeline");

    for (int r = 0; r < num_rep; ++r) {

        aprConverter.get_apr(apr_vec[r],pipelineImage);

        part_vec[r].sample_parts_from_img_downsampled(apr_vec[r],pipelineImage);

    }

    timer.stop_timer();


    //analysisData.add_timer(aprConverter.method_timer,apr.total_number_particles(),num_rep);

    analysisData.add_timer(aprConverter.computation_timer);

    //Required in all benchmarks
    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);

}



