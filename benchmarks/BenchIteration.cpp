
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

#include "APRBenchHelper.hpp"

template<typename partsType>
void bench_pixel_iteration(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);

template<typename partsType>
inline void bench_apr_iteration_old(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);

template<typename partsType>
inline void bench_apr_iteration(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);

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

        //bench_apr_iteration_old(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);

        //put benchmark funtions here..
        bench_apr_iteration(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);

        if((i==0) && !options.no_pixel){
            /*
            * Pixel benchmarks (These are content independent)
            */
            bench_pixel_iteration(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);
        }
    }
    benchAPRHelper.analysisData.init(analysis_file_name,"iteration_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();
}


template<typename partsType>
inline void bench_apr_iteration(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    APRTimer timer(true);

    auto lin_it = apr.iterator();

    std::cout << "CR: " << apr.computational_ratio() << std::endl;

    //burn in
    for (int r = 0; r < 10; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(lin_it)
#endif
            for (z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
                    for (lin_it.begin(level, z, x); lin_it < lin_it.end(); lin_it++) {
                        //need to add the ability to get y, and x,z but as possible should be lazy.

                        parts[lin_it] = (lin_it.y());
                    }
                }
            }
        }
    }
    

    timer.start_timer("iteration_y_openmp");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(lin_it)
#endif
            for (int z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
                    for (lin_it.begin(level, z, x); lin_it < lin_it.end(); lin_it++) {
                        //need to add the ability to get y, and x,z but as possible should be lazy.
                        // uint64_t idx = lin_it;
                        parts[lin_it] = (lin_it.y());
                    }
                }
            }
        }
    }
    timer.stop_timer();


    timer.start_timer("iteration_y_openmp_static");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static) firstprivate(lin_it)
#endif
            for (int z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
                    for (lin_it.begin(level, z, x); lin_it < lin_it.end(); lin_it++) {
                        //need to add the ability to get y, and x,z but as possible should be lazy.
                        // uint64_t idx = lin_it;
                        parts[lin_it] = (lin_it.y());

                    }
                }
            }
        }
    }

    timer.stop_timer();


    timer.start_timer("iteration_y");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {

            for (int z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
                    for (lin_it.begin(level, z, x); lin_it < lin_it.end(); lin_it++) {

                        parts[lin_it] = lin_it.y();
                    }
                }
            }
        }
    }

    timer.stop_timer();

    timer.start_timer("iteration_noy_openmp");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(lin_it)
#endif
            for (int z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
                    for (lin_it.begin(level, z, x); lin_it < lin_it.end(); lin_it++) {
                        //need to add the ability to get y, and x,z but as possible should be lazy.

                        parts[lin_it] += 1;

                    }
                }
            }
        }
    }

    timer.stop_timer();


    auto parent_it = apr.tree_iterator();

    timer.start_timer("iteration_parent");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            for (int z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {

                    parent_it.begin(level-1, z/2, x/2);

                    for (lin_it.begin(level, z, x); lin_it < lin_it.end(); lin_it++) {

                        while(parent_it.y() < lin_it.y()/2 && parent_it < parent_it.end()) {
                            parent_it++;
                        }
                        parts[lin_it] = parent_it.y();

                    }
                }
            }
        }
    }

    timer.stop_timer();


    timer.start_timer("iteration_parent_openmp");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_max(); level >= lin_it.level_min(); --level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(lin_it, parent_it)
#endif
            for (int z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {

                    parent_it.begin(level-1, z/2, x/2);

                    for (lin_it.begin(level, z, x); lin_it < lin_it.end(); lin_it++) {

                        while(parent_it.y() < lin_it.y()/2 && parent_it < parent_it.end()) {
                            parent_it++;
                        }
                        parts[lin_it] = parent_it.y();

                    }
                }
            }
        }
    }

    timer.stop_timer();


    timer.start_timer("iteration_parent_blocked_openmp");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(lin_it, parent_it)
#endif
            for (int z_d = 0; z_d < parent_it.z_num(level-1); z_d++) {
                for(int z = z_d*2; z <= std::min(z_d*2+1, lin_it.z_num(level)-1); ++z) {
                    for (int x_d = 0; x_d < parent_it.x_num(level - 1); ++x_d) {
                        for(int x = 2*x_d; x <= std::min(x_d*2+1, lin_it.x_num(level)-1); ++x) {

                            parent_it.begin(level - 1, z / 2, x / 2);

                            for (lin_it.begin(level, z, x); lin_it < lin_it.end(); lin_it++) {

                                while (parent_it.y() < lin_it.y() / 2 && parent_it < parent_it.end()) {
                                    parent_it++;
                                }
                                parts[lin_it] = parent_it.y();

                            }
                        }
                    }
                }
            }
        }
    }

    timer.stop_timer();

    //Required in all benchmarks
    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);

}


template<typename partsType>
inline void bench_apr_iteration_old(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    APRTimer timer(true);

    auto it = apr.random_iterator();

    std::cout << "CR: " << apr.computational_ratio() << std::endl;

    //burn in

    for (int r = 0; r < 10; ++r) {
        for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
            int z = 0;
            int x = 0;

            for (z = 0; z < it.z_num(level); z++) {
                for (x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it < it.end(); it++) {

                        parts[it] = (uint16_t)(parts[it] + 1);
                    }
                }
            }
        }
    }

    timer.start_timer("apr_old_iteration");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
            int z = 0;
            int x = 0;

            for (z = 0; z < it.z_num(level); z++) {
                for (x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it < it.end(); it++) {

                        parts[it] = (uint16_t)(parts[it] + 1);
                    }
                }
            }
        }
    }

    timer.stop_timer();

    timer.start_timer("apr_old_iteration_openmp");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
            for (int z = 0; z < it.z_num(level); z++) {
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.set_new_lzx(level, z, x); it < it.end(); it++) {

                        parts[it] = (uint16_t)(parts[it] + it.y());
                    }
                }
            }
        }
    }

    timer.stop_timer();

    //Required in all benchmarks
    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);

}

template<typename partsType>
void bench_pixel_iteration(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    APRTimer timer(true);

    PixelData<partsType> test_img;

    test_img.init(apr.org_dims(0),apr.org_dims(1),apr.org_dims(2));

    timer.start_timer("pixel_iteration");

    for (int r = 0; r < num_rep; ++r) {

        for (int z = 0; z < test_img.z_num; ++z) {
            for (int x = 0; x < test_img.x_num; ++x) {
                for (int y = 0; y < test_img.y_num; ++y) {

                    test_img.at(y,x,z) = test_img.at(y,x,z) + 1;

                }
            }
        }

    }

    timer.stop_timer();

    timer.start_timer("pixel_iteration_openmp");

    for (int r = 0; r < num_rep; ++r) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
        for (int z = 0; z < test_img.z_num; ++z) {
            for (int x = 0; x < test_img.x_num; ++x) {
                for (int y = 0; y < test_img.y_num; ++y) {
                    test_img.at(y,x,z) = (uint16_t) (test_img.at(y,x,z) + 1);
                }
            }
        }

    }

    timer.stop_timer();

    timer.start_timer("pixel_iteration_work_openmp");

    for (int r = 0; r < num_rep; ++r) {

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
        for (int z = 0; z < test_img.z_num; ++z) {
            for (int x = 0; x < test_img.x_num; ++x) {
                for (int y = 0; y < test_img.y_num; ++y) {
                    test_img.at(y,x,z) = (uint16_t) log(exp(test_img.at(y,x,z) + 1));
                }
            }
        }

    }

    timer.stop_timer();

    //Required in all benchmarks
    analysisData.add_timer(timer,test_img.mesh.size(),num_rep);

}

