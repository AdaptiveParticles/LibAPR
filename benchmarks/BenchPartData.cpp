//
// Created by cheesema on 2019-07-03.
//

//////////////////////////////////////////////////////
///
/// Bevan Cheeseman 2019
const char* usage = R"(
Benchmarking iteration performance.

Usage:

BenchPartData

)";


#include <algorithm>
#include <iostream>

#include "APRBenchHelper.hpp"

template<typename partsType>
inline void bench_particle_data(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);
template<typename partsType>
inline void bench_lazy_particle_data(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData);

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

        bench_particle_data(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);

        bench_lazy_particle_data(apr,parts,benchAPRHelper.get_number_reps(),benchAPRHelper.analysisData);

    }

    benchAPRHelper.analysisData.init(analysis_file_name,"particle_datastructure_benchmarks",argc,argv);
    benchAPRHelper.analysisData.write_analysis_data_hdf5();

}
template<typename partsType>
inline void bench_particle_data(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData) {
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    APRTimer timer(true);

    float CR = apr.computational_ratio();

    std::cout << "CR: " << CR << std::endl;

    auto lin_it = apr.iterator();

    timer.start_timer("particle_data");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z,x) firstprivate(lin_it)
#endif
            for (z = 0; z < lin_it.z_num(level); z++) {
                for (x = 0; x < lin_it.x_num(level); ++x) {
                    for (lin_it.begin(level, z, x); lin_it < lin_it.end();
                         lin_it++) {
                        //need to add the ability to get y, and x,z but as possible should be lazy.
                        parts[lin_it] += 1;

                    }
                }
            }
        }
    }

    timer.stop_timer();

    PartCellData<uint16_t> partCellData;
    partCellData.init(apr);

    timer.start_timer("partcell_data");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z,x) firstprivate(lin_it)
#endif
            for (z = 0; z < lin_it.z_num(level); z++) {
                for (x = 0; x < lin_it.x_num(level); ++x) {
                    for (lin_it.begin(level, z, x); lin_it < lin_it.end();
                         lin_it++) {
                        //need to add the ability to get y, and x,z but as possible should be lazy.

                        partCellData[lin_it] += 1;
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
inline void bench_lazy_particle_data(APR& apr,ParticleData<partsType>& parts,int num_rep,AnalysisData& analysisData) {

    APRTimer timer(true);

    auto it = apr.iterator();

    std::string file_name = "parts_lazy_bench.apr";

    APRFile writeFile;

    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(apr);

    writeFile.write_particles("parts",parts);

    std::cout << "APR File Size: " << writeFile.current_file_size_GB() << " GB" << std::endl;
    std::cout << "Original Image Size: " << (apr.org_dims(0)*apr.org_dims(1)*apr.org_dims(2)*2)/(1000000000.0) << " GB" << std::endl;

    writeFile.close();

    writeFile.open(file_name,"READWRITE");

    LazyData<uint16_t> parts_lazy;

    parts_lazy.init(writeFile, "parts");

    parts_lazy.open();

    timer.start_timer("read_write_loop_slice");
    for (int r = 0; r < num_rep; ++r) {
        for (int level = (it.level_max()); level >= it.level_min(); --level) {
            int z = 0;

#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it,parts_lazy)
#endif
            for (z = 0; z < it.z_num(level); z++) {
                parts_lazy.load_slice(level, z, it);
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it < it.end();
                         it++) {
                        //add caching https://support.hdfgroup.org/HDF5/doc/H5.user/Caching.html

                        parts_lazy[it] += 1;

                    }
                }
                parts_lazy.write_slice(level, z, it);
            }
        }
    }

    timer.stop_timer();

    timer.start_timer("read_loop_slice");

    for (int r = 0; r < num_rep; ++r) {
        for (int level = (it.level_max()); level >= it.level_min(); --level) {
            int z = 0;
            int x = 0;

            for (z = 0; z < it.z_num(level); z++) {
                parts_lazy.load_slice(level, z, it);
#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(x) firstprivate(it)
#endif
                for (x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it < it.end();
                         it++) {
                        //add caching https://support.hdfgroup.org/HDF5/doc/H5.user/Caching.html

                        parts_lazy[it] += 1;
                    }
                }
            }
        }
    }

    timer.stop_timer();


    parts_lazy.close();

    ParticleData<uint16_t> parts_read;
    timer.start_timer("normal_read");

    writeFile.read_particles(apr, "parts", parts_read);

    timer.stop_timer();

    timer.start_timer("normal_write");

    writeFile.write_particles("parts_t", parts_read);

    timer.stop_timer();

    writeFile.close();

    analysisData.add_timer(timer,apr.total_number_particles(),num_rep);


}


