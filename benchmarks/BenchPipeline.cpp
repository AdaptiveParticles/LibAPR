
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

    APRTimer timer_steps(false);

    PixelData<partsType> pipelineImage;

    APRReconstruction::reconstruct_constant(apr,pipelineImage,parts);

    APRConverter<partsType> aprConverter;

    aprConverter.par.auto_parameters = false;
    aprConverter.par.grad_th = 10;

    std::vector<APR> apr_vec;
    apr_vec.resize(num_rep);

    std::vector<ParticleData<partsType>> part_vec;
    part_vec.resize(num_rep);

    APRFile aprFile;
    std::string file_name = "bench_pipeline.apr";

    timer.start_timer("pipeline");

    aprFile.open("file_name","WRITE");

    for (int r = 0; r < num_rep; ++r) {

        timer_steps.start_timer("get_apr");

        aprConverter.get_apr(apr_vec[r],pipelineImage);

        timer_steps.stop_timer();

        timer_steps.start_timer("sample_parts");

        part_vec[r].sample_image(apr_vec[r], pipelineImage);

        timer_steps.stop_timer();

        timer_steps.start_timer("write_apr");

        aprFile.write_apr(apr_vec[r], r, "t", false);

        timer_steps.stop_timer();

        timer_steps.start_timer("write_parts");

        aprFile.write_particles("parts",part_vec[r],true,r);

        timer_steps.stop_timer();

    }

    timer.stop_timer();

    aprFile.close();

    analysisData.add_timer_avg(aprConverter.computation_timer);

    analysisData.add_timer_avg(aprConverter.method_timer);

    analysisData.add_timer_avg(timer_steps);

    //Required in all benchmarks
    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);

}



