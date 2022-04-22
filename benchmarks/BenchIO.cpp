
//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2019
const char* usage = R"(
Benchmarking iteration performance.

Usage:

BenchIO

)";


#include <algorithm>
#include <iostream>

#include "APRBenchHelper.hpp"

template<typename partsType>
void bench_apr_io(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);

template<typename partsType>
void bench_pixel_io(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);

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
        bench_apr_io(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);

        if((i==0) && !options.no_pixel){
            bench_pixel_io(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);
        }

    }


    benchAPRHelper.analysisData.init(analysis_file_name,"iteration_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}

template<typename partsType>
void bench_pixel_io(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData){

    APRTimer timer(true);

    PixelData<uint16_t> img;

    img.init(apr.org_dims(0),apr.org_dims(1),apr.org_dims(2));

    timer.start_timer("write_tiff");
    for (int r = 0; r < num_rep; ++r) {
        TiffUtils::saveMeshAsTiff("pixel_test.tif",img,false);

    }
    timer.stop_timer();

    timer.start_timer("read_tiff");
    for (int r = 0; r < num_rep; ++r) {
        PixelData<uint16_t> temp_img = TiffUtils::getMesh<uint16_t>("pixel_test.tif");

    }
    timer.stop_timer();

    analysisData.add_timer(timer,img.mesh.size(),num_rep);

}

template<typename partsType>
inline void bench_apr_io(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    APRTimer timer(true);

    APRTimer timer_steps(false);

    APRFile aprFile;
    const std::string file_name = "bench_pipeline.apr";

    timer.start_timer("write");

    aprFile.open(file_name,"WRITE");

    for (int r = 0; r < num_rep; ++r) {

        timer_steps.start_timer("write_apr");

        aprFile.write_apr(apr, r, "t", false);

        timer_steps.stop_timer();

        timer_steps.start_timer("write_parts");

        aprFile.write_particles("parts",parts,true,r);

        timer_steps.stop_timer();

    }

    timer.stop_timer();

    aprFile.close();

    timer.start_timer("read");

    aprFile.open(file_name,"READ");

    for (int r = 0; r < num_rep; ++r) {

        timer_steps.start_timer("read_apr");

        APR apr;
        ParticleData<uint16_t> parts;

        aprFile.read_apr(apr,r);

        timer_steps.stop_timer();

        timer_steps.start_timer("read_parts");

        aprFile.read_particles(apr,"parts",parts,true,r);

        timer_steps.stop_timer();

    }

    timer.stop_timer();

    aprFile.close();


    analysisData.add_timer_avg(timer_steps);

    analysisData.add_timer_avg(aprFile.timer);

    //Required in all benchmarks
    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);

}



