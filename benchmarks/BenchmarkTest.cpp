//
// Created by cheesema on 21.01.18.
//

#include <gtest/gtest.h>
#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRConverter.hpp"
#include <utility>
#include <cmath>
#include "../test/TestTools.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include "io/APRWriter.hpp"
#include "io/APRFile.hpp"

#include "data_structures/APR/particles/LazyData.hpp"


class BenchmarkAPR {


public:
    /**
     * tileAPR - tiles an APR to generate a larger APR for testing purposes.
     * @param tile_dims a 3 dimensional vector with the number of times in each direction to tile
     * @param apr_input The APR to be tiled
     * @tparam parts particles to be tiled
     * @param apr_tiled The tiled APR
     * @tparam tiled_parts The tiled particles
*/
    template<typename T>
    static void tileAPR(std::vector<int> tile_dims, APR& apr_input,ParticleData<T>& parts, APR& apr_tiled,ParticleData<T>& tiled_parts) {
        //
        // Note the data-set should be a power of 2 in its dimensons for this to work.
        //


        APRTimer timer(true);

        if (tile_dims.size() != 3) {
            std::cerr << "in-correct tilling dimensions" << std::endl;
        }

        auto it = apr_input.iterator();

        float check_y = log2(1.0f*it.org_dims(0));
        float check_x = log2(1.0f*it.org_dims(1));
        float check_z = log2(1.0f*it.org_dims(2));

        //this function only works for datasets that are powers of 2.
        bool pow_2y = (check_y - std::floor(check_y)) == 0;
        bool pow_2x = (check_x - std::floor(check_x)) == 0;
        bool pow_2z = (check_z - std::floor(check_z)) == 0;

        if(!pow_2y || !pow_2x || !pow_2z){
            std::cerr << "The dimensions must be a power of 2!" << std::endl;
        }

        //round up to nearest even dimension



        int org_dims_y = std::ceil(it.org_dims(0)/2.0f)*2;
        int org_dims_x = std::ceil(it.org_dims(1)/2.0f)*2;
        int org_dims_z = std::ceil(it.org_dims(2)/2.0f)*2;


        //now to tile the APR
        auto new_y_num = org_dims_y * tile_dims[0];
        auto new_x_num = org_dims_x * tile_dims[1];
        auto new_z_num = org_dims_z * tile_dims[2];


        apr_tiled.aprInfo.init(new_y_num, new_x_num, new_z_num);
        apr_tiled.linearAccess.genInfo = &apr_tiled.aprInfo;

        PullingSchemeSparse ps;
        ps.initialize_particle_cell_tree(apr_tiled.aprInfo);


        timer.start_timer("init sparse tree");

        const int level_offset = apr_tiled.level_max() - apr_input.level_max();

        for (int z_tile = 0; z_tile < tile_dims[2]; ++z_tile) {
            for (int x_tile = 0; x_tile < tile_dims[1]; ++x_tile) {
                for (int y_tile = 0; y_tile < tile_dims[0]; ++y_tile) {

                    for (int level = it.level_min(); level <= it.level_max(); level++) {
                        int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it)
#endif
                        for (z = 0; z < it.z_num(level); z++) {
                            for (int x = 0; x < it.x_num(level); ++x) {

                                int new_level = level_offset + level;

                                //transform the co-ordinates to the new frame of reference..

                                int z_offset = (z_tile * org_dims_z) / apr_tiled.aprInfo.level_size[new_level];
                                int x_offset = (x_tile * org_dims_x) / apr_tiled.aprInfo.level_size[new_level];
                                int y_offset = (y_tile * org_dims_y) / apr_tiled.aprInfo.level_size[new_level];


                                if (level == it.level_max()) {
                                    if ((x % 2 == 0) && (z % 2 == 0)) {
                                        const size_t offset_pc =
                                                apr_tiled.aprInfo.x_num[new_level - 1] * ((z + z_offset) / 2) +
                                                ((x + x_offset) / 2);

                                        auto &mesh = ps.particle_cell_tree.data[new_level - 1][offset_pc][0].mesh;

                                        for (it.begin(level, z, x); it < it.end();
                                             it++) {
                                            //insert
                                            int y = (it.y() + y_offset) / 2;
                                            mesh[y] = 1;
                                        }
                                    }
                                } else {

                                    const size_t offset_pc =
                                            apr_tiled.aprInfo.x_num[new_level] * (z + z_offset) + (x + x_offset);

                                    auto &mesh = ps.particle_cell_tree.data[new_level][offset_pc][0].mesh;

                                    for (it.begin(level, z, x); it < it.end();
                                         it++) {
                                        //insert
                                        mesh[it.y() + y_offset] = 4;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        timer.stop_timer();

        timer.start_timer("pulling scheme");

        //ps.pulling_scheme_main();

        timer.stop_timer();

        timer.start_timer("initialize structures");

        apr_tiled.linearAccess.initialize_linear_structure_sparse(apr_input.parameters, ps.particle_cell_tree);

        apr_tiled.apr_initialized = true;

        timer.stop_timer();

        tiled_parts.init(apr_tiled.total_number_particles());

        auto it_tile = apr_tiled.iterator();


        //auto it_tree = apr_input.tree_iterator();

        ParticleData<float> tparts_input;

        APRTreeNumerics::fill_tree_mean(apr_input, parts, tparts_input);

        //
        // Now we need to sample the particles, with the fact that some particles with have increased resolution in the merge
        // potentially due to boundary effects
        //

        int org_y_num = org_dims_y;
        int org_x_num = org_dims_x;
        int org_z_num = org_dims_z;

        //now compute the particles
        for (int level = (it.level_min() + level_offset); level <= it_tile.level_max(); level++) {

            auto level_org = level - level_offset;
            int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it,it_tile)
#endif
            for (z = 0; z < it_tile.z_num(level); z++) {

                int z_org =
                        (((z) * apr_tiled.aprInfo.level_size[level]) % org_z_num) / apr_tiled.aprInfo.level_size[level];

                for (int x = 0; x < it_tile.x_num(level); ++x) {

                    int x_org = (((x) * apr_tiled.aprInfo.level_size[level]) % org_x_num) /
                                apr_tiled.aprInfo.level_size[level];


                    it.begin(level_org, z_org, x_org);


                    for (it_tile.begin(level, z, x); it_tile < it_tile.end(); ++it_tile) {

                        int y_org = (((it_tile.y()) * apr_tiled.aprInfo.level_size[level]) % org_y_num) /
                                    apr_tiled.aprInfo.level_size[level];


                        while ((it.y() < y_org) && (it < it.end())) {
                            it++;
                        }


                        if ((it.y() == y_org) && (it != it.end())) {
                            tiled_parts[it_tile] = parts[it];
                        }

                        if (it >= (it.end() - 1)) {
                            it.begin(level_org, z_org, x_org);
                        }

                    }
                }
            }
        }

    };




};

struct TestData{

    APR apr;
    PixelData<uint16_t> img_level;
    PixelData<uint16_t> img_type;
    PixelData<uint16_t> img_original;
    PixelData<uint16_t> img_pc;
    PixelData<uint16_t> img_x;
    PixelData<uint16_t> img_y;
    PixelData<uint16_t> img_z;

    ParticleData<uint16_t> particles_intensities;

    std::string filename;
    std::string apr_filename;
    std::string output_name;
    std::string output_dir;

};

struct BenchmarkData{

    std::vector<APR> aprs;
    std::vector<ParticleData<uint16_t>> parts;


};

class CreateAPRTest : public ::testing::Test {
public:

    TestData test_data;
    BenchmarkData bench_data;

protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

};

class CreateBenchmarkAPR : public CreateAPRTest
{
public:
    void SetUp() override;
};


class CreateSmallSphereTest : public CreateAPRTest
{
public:
    void SetUp() override;
};


class Create210SphereTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateGTSmallTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateGTSmall2DTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class CreateGTSmall1DTest : public CreateAPRTest
{
public:
    void SetUp() override;
};


bool test_tiling(BenchmarkData& benchmarkData){

    int apr_num = 0;

    std::vector<int> tile_dims = {10,10,10};

    APR apr_tiled;
    ParticleData<uint16_t> tiled_parts;

    APRTimer timer(true);

    timer.start_timer("tile APR");

    BenchmarkAPR::tileAPR(tile_dims,benchmarkData.aprs[apr_num], benchmarkData.parts[apr_num],apr_tiled,tiled_parts);

    timer.stop_timer();

    std::cout << "Original number particles: " << benchmarkData.aprs[apr_num].total_number_particles() << std::endl;
    std::cout << "Tiled number particles: " << apr_tiled.total_number_particles() << std::endl;

    std::cout << "Original size: " << benchmarkData.aprs[apr_num].org_dims(0) << "x" << benchmarkData.aprs[apr_num].org_dims(1) << "x" << benchmarkData.aprs[apr_num].org_dims(2) << std::endl;
    std::cout << "New size: " << apr_tiled.org_dims(0) << "x" << apr_tiled.org_dims(1) << "x" << apr_tiled.org_dims(2) << std::endl;

    std::cout << "Old CR: " << benchmarkData.aprs[apr_num].computational_ratio() << std::endl;
    std::cout << "Tiled CR: " << apr_tiled.computational_ratio() << std::endl;


    timer.start_timer("init tree");
    auto it_t = apr_tiled.tree_iterator();
    (void) it_t;
    timer.stop_timer();

    APRFile aprFile;
    aprFile.timer.verbose_flag = true;

    aprFile.open("tiledAPR.apr","WRITE");
    aprFile.write_apr(apr_tiled);
    aprFile.write_particles(apr_tiled,"particle_intensities",tiled_parts);

    std::cout << "APR File Size: " << aprFile.current_file_size_GB() << " GB" << std::endl;
    std::cout << "Original Image Size: " << (apr_tiled.org_dims(0)*apr_tiled.org_dims(1)*apr_tiled.org_dims(2)*2)/(1000000000.0) << " GB" << std::endl;

    aprFile.close();

    APR aprRead;
    aprFile.open("tiledAPR.apr","READ");

    timer.start_timer("read_apr");
    aprFile.read_apr(aprRead);
    timer.stop_timer();

    ParticleData<uint16_t> parts;
    timer.start_timer("read_parts");

    aprFile.read_particles(aprRead,"particle_intensities",parts);

    timer.stop_timer();
    aprFile.close();


    return true;
}

void bench_io(){

}

void bench_compress(){

}

void bench_lazy_particles(BenchmarkData& benchmarkData){


    int apr_num = 0;

    std::vector<int> tile_dims = {10,10,10};

    APR apr_tiled;
    ParticleData<uint16_t> tiled_parts;

    APRTimer timer(true);

    timer.start_timer("tile APR");

    BenchmarkAPR::tileAPR(tile_dims,benchmarkData.aprs[apr_num], benchmarkData.parts[apr_num],apr_tiled,tiled_parts);

    timer.stop_timer();

    auto it = apr_tiled.iterator();

    std::string file_name = "parts_lazy_bench.apr";

    APRFile writeFile;

    writeFile.open(file_name,"WRITE");

    writeFile.write_apr(apr_tiled);

    writeFile.write_particles(apr_tiled,"parts",tiled_parts);


    std::cout << "APR File Size: " << writeFile.current_file_size_GB() << " GB" << std::endl;
    std::cout << "Original Image Size: " << (apr_tiled.org_dims(0)*apr_tiled.org_dims(1)*apr_tiled.org_dims(2)*2)/(1000000000.0) << " GB" << std::endl;

    writeFile.close();

    writeFile.open(file_name,"READWRITE");

    LazyData<uint16_t> parts_lazy;

    parts_lazy.init_file(writeFile,"parts",true);

    parts_lazy.open();

    timer.start_timer("read write loop slice");

    for (int level = (it.level_max()); level >= it.level_min(); --level) {
        int z = 0;

#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it,parts_lazy)
#endif
        for (z = 0; z < it.z_num(level); z++) {
            parts_lazy.load_slice(level,z,it);
            for (int x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level,z,x); it < it.end();
                     it++) {
                    //add caching https://support.hdfgroup.org/HDF5/doc/H5.user/Caching.html

                    parts_lazy[it] += 1;

                }
            }
            parts_lazy.write_slice(level,z,it);
        }
    }

    timer.stop_timer();

    float read_write_lazy = timer.timings.back();

    timer.start_timer("read loop slice");

    for (int level = (it.level_max()); level >= it.level_min(); --level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            parts_lazy.load_slice(level,z,it);
#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(x) firstprivate(it)
#endif
            for (x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level,z,x); it < it.end();
                     it++) {
                    //add caching https://support.hdfgroup.org/HDF5/doc/H5.user/Caching.html

                    parts_lazy[it] += 1;
                }
            }
        }
    }

    timer.stop_timer();

    float read_lazy = timer.timings.back();

    parts_lazy.close();


    timer.start_timer("normal");

    for (int level = (it.level_max()); level >= it.level_min(); --level) {
        int z = 0;
        //int x = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it)
#endif
        for (z = 0; z < it.z_num(level); z++) {
            for (int x = 0; x < it.x_num(level); ++x) {
                for (it.begin(level,z,x); it < it.end();
                     it++) {

                    tiled_parts[it] += 1;
                }
            }
        }
    }

    timer.stop_timer();



    float normal_iterate = timer.timings.back();

    timer.start_timer("normal read");
    ParticleData<uint16_t> parts_read;
    writeFile.read_particles(apr_tiled,"parts",parts_read);
    timer.stop_timer();

    float normal_read = timer.timings.back();

    timer.start_timer("normal write");
    writeFile.write_particles(apr_tiled,"parts_t",parts_read);
    timer.stop_timer();

    float normal_write = timer.timings.back();

    std::cout << "Read and loop: " << normal_read + normal_iterate << std::endl;
    std::cout << "Lazy read loop: " << read_lazy << std::endl;

    std::cout << "Read, loop, and write: " << normal_read + normal_iterate + normal_write << std::endl;
    std::cout << "Lazy io loop: " << read_write_lazy << std::endl;

    writeFile.close();


}


bool bench_particle_structures(BenchmarkData& benchmarkData) {
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    int apr_num = 0;

    int full_size = 1024;
    int dim_sz = full_size/benchmarkData.aprs[apr_num].org_dims(0);


    std::vector<int> tile_dims = {dim_sz,dim_sz,dim_sz};

    APR apr_tiled;
    ParticleData<uint16_t> parts;

    APRTimer timer(true);

    timer.start_timer("tile APR");

    BenchmarkAPR::tileAPR(tile_dims,benchmarkData.aprs[apr_num], benchmarkData.parts[apr_num],apr_tiled,parts);

    bool success = true;

    float CR = apr_tiled.computational_ratio();

    std::cout << "CR: " << CR << std::endl;

    unsigned int num_rep = 1000;

    auto lin_it = apr_tiled.iterator();

    timer.start_timer("LinearIteration - normal - OpenMP");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(lin_it)
#endif
            for (z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
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
    partCellData.init(apr_tiled);

    timer.start_timer("LinearIteration - PartCell - OpenMP (new)");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(lin_it)
#endif
            for (z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
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

    return success;

}


bool bench_iteration(BenchmarkData& benchmarkData){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    int apr_num = 0;

    int full_size = 512;
    int dim_sz = full_size/benchmarkData.aprs[apr_num].org_dims(0);

    std::vector<int> tile_dims = {dim_sz,dim_sz,dim_sz};

    APR apr_tiled;
    ParticleData<uint16_t> parts;

    APRTimer timer(true);

    timer.start_timer("tile APR");

    BenchmarkAPR::tileAPR(tile_dims,benchmarkData.aprs[apr_num], benchmarkData.parts[apr_num],apr_tiled,parts);

    bool success = true;

    auto it = apr_tiled.random_iterator();

    PixelData<uint16_t> test_img;

    int img_size = 512;
    float pix_scale = pow(img_size,3);

    float apr_scale = pow(full_size,3);

    test_img.init(img_size,img_size,img_size);

    float CR = test_img.mesh.size()/(1.0f*it.total_number_particles());

    std::cout << "CR: " << CR << std::endl;

    unsigned int num_rep = 100;

    //Add + 1 to the value, while having access to (x,y,z) test;

    for (int r = 0; r < num_rep; ++r) {

        for (int z = 0; z < test_img.z_num; ++z) {
            for (int x = 0; x < test_img.x_num; ++x) {
                for (int y = 0; y < test_img.y_num; ++y) {

                    test_img.at(y,x,z) = test_img.at(y,x,z) + 1;

                }
            }
        }
    }


    timer.start_timer("Pixel Iteration - Serial");

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

    timer.start_timer("Pixel Iteration - OpenMP");

    //int z = 0;

    for (int r = 0; r < num_rep; ++r) {
        int z = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for private(z)
#endif
        for (z = 0; z < test_img.z_num; ++z) {
            for (int x = 0; x < test_img.x_num; ++x) {
                for (int y = 0; y < test_img.y_num; ++y) {
                    test_img.at(y,x,z) = (uint16_t) (test_img.at(y,x,z) + 1);
                }
            }
        }

    }

    timer.stop_timer();

    auto mesh_it = timer.timings.back();

    timer.start_timer("APR Iteration - Serial");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
            int z = 0;
            int x = 0;

            for (z = 0; z < it.z_num(level); z++) {
                for (x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it < it.end();
                         it++) {
                        parts[it] = (uint16_t)(parts[it] + 1);

                    }
                }
            }
        }
    }

    timer.stop_timer();


    timer.start_timer("APR Iteration - OpenMP");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it)
#endif
            for (z = 0; z < it.z_num(level); z++) {
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.set_new_lzx(level, z, x); it < it.end();
                         it++) {
                        parts[it] = (uint16_t)(parts[it] + 1);

                    }
                }
            }
        }
    }

    timer.stop_timer();

    auto org_it = timer.timings.back();


    auto lin_it = apr_tiled.iterator();

    timer.start_timer("LinearIteration (inc y) - OpenMP");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(lin_it)
#endif
            for (z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
                    for (lin_it.begin(level, z, x); lin_it < lin_it.end();
                         lin_it++) {
                        //need to add the ability to get y, and x,z but as possible should be lazy.
                        parts[lin_it] = (lin_it.y());

                    }
                }
            }
        }
    }

    timer.stop_timer();

    auto lin_time = timer.timings.back();


    timer.start_timer("LinearIteration (without y) - OpenMP");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(lin_it)
#endif
            for (z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
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

    auto lin_time_noy = timer.timings.back();

    mesh_it = mesh_it*pix_scale;
    org_it *= apr_scale;
    lin_time*= apr_scale;
    lin_time_noy *=apr_scale;

    std::cout << "SU (old): " << mesh_it/org_it << std::endl;
    std::cout << "SU (linear no y): " << mesh_it/lin_time_noy << std::endl;
    std::cout << "SU (linear): " << mesh_it/lin_time << std::endl;
    std::cout << "SU vs old: " << org_it/lin_time << std::endl;

    return success;
}


bool bench_pipeline(TestData& test_data,float rel_error){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    bool success = true;

    //the apr datastructure
    APR apr;
    APRConverter<uint16_t> aprConverter;

    //read in the command line options into the parameters file
    aprConverter.par.Ip_th = 0;
    aprConverter.par.rel_error = rel_error;
    aprConverter.par.lambda = 0;
    aprConverter.par.mask_file = "";
    aprConverter.par.min_signal = -1;

    aprConverter.par.sigma_th_max = 50;
    aprConverter.par.sigma_th = 100;

    aprConverter.par.SNR_min = -1;

    aprConverter.par.auto_parameters = false;

    aprConverter.par.output_steps = true;

    //where things are
    aprConverter.par.input_image_name = test_data.filename;
    aprConverter.par.input_dir = "";
    aprConverter.par.name = test_data.output_name;
    aprConverter.par.output_dir = test_data.output_dir;

    //Gets the APR

    ParticleData<uint16_t> particles_intensities;

    APRTimer timer(true);

    timer.start_timer("APR Structures");

    aprConverter.get_apr(apr,test_data.img_original);

    timer.stop_timer();

    timer.start_timer("sample particles");

    particles_intensities.sample_parts_from_img_downsampled(apr,test_data.img_original);

    timer.stop_timer();


    return success;
}

std::string get_source_directory_apr(){
    // returns path to the directory where utils.cpp is stored

    std::string tests_directory = std::string(__FILE__);
    tests_directory = tests_directory.substr(0, tests_directory.find_last_of("\\/") + 1);

    return tests_directory;
}

void CreateBenchmarkAPR::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/cr_20.apr";
    bench_data.aprs.resize(1);
    bench_data.parts.resize(1);

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(bench_data.aprs[0]);
    aprFile.read_particles(bench_data.aprs[0],"particle_intensities",bench_data.parts[0]);
    aprFile.close();

}


TEST_F(CreateBenchmarkAPR, BENCH_ITERATION) {

    //ASSERT_TRUE(bench_iteration(bench_data));

}

TEST_F(CreateBenchmarkAPR, BENCH_STRUCTURES) {

    //ASSERT_TRUE(bench_particle_structures(bench_data));

}

TEST_F(CreateBenchmarkAPR, BENCH_TILE) {

    //ASSERT_TRUE(test_tiling(bench_data));

}

TEST_F(CreateBenchmarkAPR, BENCH_LAZY_PARTS) {

    bench_lazy_particles(bench_data);

}




int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}

