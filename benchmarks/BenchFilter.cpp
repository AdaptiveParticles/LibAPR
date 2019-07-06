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


#include <algorithm>
#include <iostream>

#include "BenchAPRHelper.hpp"
#include "numerics/APRFilter.hpp"

template<typename partsType>
inline void bench_apr_convolve(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size = 3);

template<typename partsType>
inline void bench_pixel_convolve(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size);

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

        //put benchmark funtions here..
        bench_apr_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,1);
        bench_apr_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,3);
        bench_apr_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,5);

        if(i==0){
            /*
            * Pixel benchmarks (These are content independent)
            */

            bench_pixel_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,1);
            bench_pixel_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,3);
            bench_pixel_convolve(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData,5);

        }
    }


    benchAPRHelper.analysisData.init(analysis_file_name,"filter_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}

template<typename partsType>
inline void bench_apr_convolve(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size){

    APRTimer timer(true);

    std::vector<PixelData<float>> stencils;
    stencils.resize(1);

    auto it = apr.iterator();

    if(it.number_dimensions() ==3){
        stencils[0].init(stencil_size, stencil_size, stencil_size);
    } else if (it.number_dimensions() ==2){
        stencils[0].init(stencil_size, stencil_size, 1);
    } else if (it.number_dimensions() ==1){
        stencils[0].init(stencil_size, 1, 1);
    }

    // unique stencil elements
    float sum = 0;
    for(int i = 0; i < stencils[0].mesh.size(); ++i) {
        sum += i;
    }
    for(int i = 0; i < stencils[0].mesh.size(); ++i) {
        stencils[0].mesh[i] = ((float) i) / sum;
    }

    APRFilter filterfns;
    filterfns.boundary_cond = false;

    timer.start_timer("apr_filter" + std::to_string(stencil_size));
    for (int r = 0; r < num_rep; ++r) {
        ParticleData<double> output;
        filterfns.convolve(apr, stencils, parts, output);
    }
    timer.stop_timer();

    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);
}
template<typename partsType>
inline void bench_pixel_convolve(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData,int stencil_size){


    std::vector<PixelData<float>> stencils;
    stencils.resize(1);

    auto it = apr.iterator();

    if(it.number_dimensions() ==3){
        stencils[0].init(stencil_size, stencil_size, stencil_size);
    } else if (it.number_dimensions() ==2){
        stencils[0].init(stencil_size, stencil_size, 1);
    } else if (it.number_dimensions() ==1){
        stencils[0].init(stencil_size, 1, 1);
    }

    // unique stencil elements
    float sum = 0;
    for(int i = 0; i < stencils[0].mesh.size(); ++i) {
        sum += i;
    }
    for(int i = 0; i < stencils[0].mesh.size(); ++i) {
        stencils[0].mesh[i] = ((float) i) / sum;
    }

    const std::vector<int> stencil_shape = {(int) stencils[0].y_num,
                                            (int) stencils[0].x_num,
                                            (int) stencils[0].z_num};

    APRTimer timer(true);

    PixelData<partsType> test_img;
    PixelData<partsType> test_img_output;

    test_img.init(apr.org_dims(0),apr.org_dims(1),apr.org_dims(2));
    test_img_output.init(test_img);

    timer.start_timer("pixel_filter" + std::to_string(stencil_size));

    const int s_plus = std::floor(stencil_size/2.0);
    const int s_minus = std::ceil(stencil_size/2.0);

    //int z = 0;

    for (int r = 0; r < num_rep; ++r) {
        int z = 0;

        const uint64_t x_num = test_img.x_num;
        const uint64_t y_num = test_img.y_num;
        const uint64_t z_num = test_img.z_num;

#ifdef HAVE_OPENMP
#pragma omp parallel for private(z)
#endif
        for (z = 0; z < test_img.z_num; ++z) {

            const int offset_max_dim3 = std::min( z + s_plus, (int) (z_num ));
            const int dim3 = std::max(z - s_minus,(int) 0);

            for (int x = 0; x < test_img.x_num; ++x) {

                const int offset_max_dim2 = std::min(x + s_plus, (int) (x_num ));
                const int dim2 = std::max(x - s_minus,(int) 0);

                for (int y = 0; y < test_img.y_num; ++y) {

                    float temp_int=0;

                    const int dim1 = std::max(y - s_minus,(int) 0);
                    const int offset_max_dim1 = std::min(y + s_plus, (int) (y_num ));

                    int counter = stencils[0].mesh.size() - 1;

                    for (int64_t q = dim3; q < offset_max_dim3; ++q) {
                        for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                            const auto off = (k) * y_num + q * y_num*x_num;
                            for (int64_t i = dim1; i < offset_max_dim1; ++i) {

                                temp_int += stencils[0].mesh[counter]*test_img.mesh[i + off];
                                counter--;
                            }
                        }
                    }

                    test_img_output.at(y,x,z) = temp_int;

                }
            }
        }

    }

    timer.stop_timer();

    //Required in all benchmarks
    analysisData.add_timer(timer,test_img.mesh.size(),num_rep);


}

