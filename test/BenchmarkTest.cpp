//
// Created by cheesema on 21.01.18.
//

#include <gtest/gtest.h>
#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRConverter.hpp"
#include <utility>
#include <cmath>
#include "TestTools.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include "io/APRWriter.hpp"
#include "io/APRFile.hpp"


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

        //round up to nearest even dimension

        auto it = apr_input.iterator();

        int org_dims_y = std::ceil(it.org_dims(0)/2.0f)*2;
        int org_dims_x = std::ceil(it.org_dims(1)/2.0f)*2;
        int org_dims_z = std::ceil(it.org_dims(2)/2.0f)*2;

//        org_dims_y = it.org_dims(0);
//        org_dims_x = it.org_dims(1);
//        org_dims_z = it.org_dims(2);





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
//#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it)
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

//                                    const size_t offset_pc =
//                                            apr_tiled.aprInfo.x_num[new_level] * (z + z_offset) + (x + x_offset);
//
//                                    auto &mesh = ps.particle_cell_tree.data[new_level][offset_pc][0].mesh;
//
//                                    for (it.begin(level, z, x); it < it.end();
//                                         it++) {
//                                        //insert
//                                        //mesh[it.y() + y_offset] = 3;
//                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        timer.stop_timer();

        timer.start_timer("pulling scheme");

        ps.pulling_scheme_main();

        timer.stop_timer();

        timer.start_timer("initialize structures");

        apr_tiled.linearAccess.initialize_linear_structure_sparse(apr_input.parameters, ps.particle_cell_tree);

        apr_tiled.apr_initialized = true;

        timer.stop_timer();

        tiled_parts.init(apr_tiled.total_number_particles());

        auto it_tile = apr_tiled.iterator();


        auto it_tree = apr_input.tree_iterator();

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
        for (int level = it_tile.level_min(); level <= it_tile.level_max(); level++) {

            const auto not_max = level < it_tile.level_max();

            auto level_org = level - level_offset;
            int z = 0;
#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it,it_tile)
#endif
            for (z = 0; z < it_tile.z_num(level); z++) {

                int z_org =
                        (((z) * apr_tiled.aprInfo.level_size[level]) % org_z_num) / apr_tiled.aprInfo.level_size[level];
                //std::cout << "level: " << level << " level org: " << level_org << " z: "  << z <<  " z_org: " << z_org << std::endl;

                for (int x = 0; x < it_tile.x_num(level); ++x) {

                    int x_org = (((x) * apr_tiled.aprInfo.level_size[level]) % org_x_num) /
                                apr_tiled.aprInfo.level_size[level];
                     //std::cout << "level: " << level << " level org: " << level_org << " x: "  << x <<  " x_org: " << x_org << std::endl;

                    it.begin(level_org, z_org, x_org);

                    if (not_max) {
                        it_tree.begin(level_org, z_org, x_org);
                    }

                    //std::cout << "Number particles: " << it.end() - it << std::endl;

                    for (it_tile.begin(level, z, x); it_tile < it_tile.end(); ++it_tile) {

                        int y_org = (((it_tile.y()) * apr_tiled.aprInfo.level_size[level]) % org_y_num) /
                                    apr_tiled.aprInfo.level_size[level];


                        while ((it.y() < y_org) && (it < it.end())) {
                            it++;
                        }

                        //std::cout << "y_g: " << it_tile.y() << " y_it: "  << it.y() <<  " y_org: " << y_org << std::endl;

                        if ((it.y() == y_org) && (it != it.end())) {
                            tiled_parts[it_tile] = parts[it];
                        }

                        if (it >= (it.end() - 1)) {
                            it.begin(level_org, z_org, x_org);
                        }


                        if (not_max) {

                            while ((it_tree.y() < y_org) && (it_tree < it_tree.end())) {
                                it_tree++;
                            }

                            if ((it_tree.y() == y_org) && it_tree != it_tree.end()) {

                                tiled_parts[it_tile] = tparts_input[it_tree];
                            }

                            //std::cout << "y_g: " << it_tile.y() << " y_it: "  << it.y() <<  " y_org: " << y_org << std::endl;

                            if (it_tree >= (it_tree.end() - 1)) {
                                it_tree.begin(level_org, z_org, x_org);
                            }
                        }
                    }
                }
            }
        }


        auto it_tile_tree = apr_tiled.tree_iterator();
        ParticleData<float> tree_parts_tiled;
        tree_parts_tiled.data.resize(apr_tiled.total_number_tree_particles());

        //now compute the particles
        for (int level = (it_tile_tree.level_min() + 1); level <= it_tile_tree.level_max(); level++) {

            auto level_org = level - level_offset;
            int z = 0;
#ifdef HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) private(z) firstprivate(it,it_tile)
#endif
            for (z = 0; z < it_tile_tree.z_num(level); z++) {

                int z_org =
                        (((z) * apr_tiled.aprInfo.level_size[level]) % org_z_num) / apr_tiled.aprInfo.level_size[level];
                //std::cout << "level: " << level << " level org: " << level_org << " z: "  << z <<  " z_org: " << z_org << std::endl;

                for (int x = 0; x < it_tile_tree.x_num(level); ++x) {

                    int x_org = (((x) * apr_tiled.aprInfo.level_size[level]) % org_x_num) /
                                apr_tiled.aprInfo.level_size[level];
                    // std::cout << "level: " << level << " level org: " << level_org << " x: "  << x <<  " x_org: " << x_org << std::endl;

                    it.begin(level_org, z_org, x_org);

                    it_tree.begin(level_org, z_org, x_org);


                    for (it_tile_tree.begin(level, z, x); it_tile_tree < it_tile_tree.end(); ++it_tile_tree) {

                        int y_org = (((it_tile_tree.y()) * apr_tiled.aprInfo.level_size[level]) % org_y_num) /
                                    apr_tiled.aprInfo.level_size[level];


                        while ((it.y() < y_org) && (it < it.end())) {
                            it++;
                        }

                        if ((it.y() == y_org) && (it != it.end())) {
                            tree_parts_tiled[it_tile_tree] = parts[it];
                        }

                        if (it >= (it.end() - 1)) {
                            it.begin(level_org, z_org, x_org);
                        }


                        while ((it_tree.y() < y_org) && (it_tree < it_tree.end())) {
                            it_tree++;
                        }

                        if ((it_tree.y() == y_org) && (it_tree != it_tree.end())) {
                            tree_parts_tiled[it_tile_tree] = tparts_input[it_tree];
                        }

                        if (it_tree >= (it_tree.end() - 1)) {
                            it_tree.begin(level_org, z_org, x_org);
                        }

                    }
                }
            }
        }


        //now push down tree
        auto it_tile_tree_up = apr_tiled.tree_iterator();

        for (int level = (it_tile_tree.level_min() + 1); level <= it_tile_tree.level_max(); ++level) {
            for (int z = 0; z < it_tile_tree.z_num(level); z++) {
                for (int x = 0; x < it_tile_tree.x_num(level); ++x) {

                    it_tile_tree_up.begin(level-1,z/2,x/2);

                    for (it_tile_tree.begin(level, z, x); it_tile_tree < it_tile_tree.end();
                         it_tile_tree++) {

                        while((it_tile_tree_up.y() < it_tile_tree.y()/2) && (it_tile_tree_up < it_tile_tree_up.end())){
                            it_tile_tree_up++;
                        }

                        if ((it_tile_tree.y()/2 == it_tile_tree_up.y()) && (it_tile_tree_up != it_tile_tree_up.end())) {
                            if (tree_parts_tiled[it_tile_tree] == 0) {

                                tree_parts_tiled[it_tile_tree] = tree_parts_tiled[it_tile_tree_up];
                            }
                        }
                    }
                }
            }
        }

        //now push down particles

        for (int level = (it_tile.level_min() + 1); level <= it_tile.level_max(); ++level) {
            for (int z = 0; z < it_tile.z_num(level); z++) {
                for (int x = 0; x < it_tile.x_num(level); ++x) {

                    it_tile_tree_up.begin(level-1,z/2,x/2);

                    for (it_tile.begin(level, z, x); it_tile < it_tile.end();
                         it_tile++) {

                        while((it_tile_tree_up.y() < it_tile.y()/2) && (it_tile_tree_up < it_tile_tree_up.end())){
                            it_tile_tree_up++;
                        }

                        if ((it_tile.y()/2 == it_tile_tree_up.y()) && (it_tile_tree_up != it_tile_tree_up.end())) {
                            if (tiled_parts[it_tile] == 0) {

                                tiled_parts[it_tile] = tree_parts_tiled[it_tile_tree_up];
                            }
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

class CreateAPRTest : public ::testing::Test {
public:

    TestData test_data;

protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

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


bool test_tiling(TestData& test_data){

    std::vector<int> tile_dims = {4,3,2};

    APR apr_tiled;
    ParticleData<uint16_t> tiled_parts;

    APRTimer timer(true);

    timer.start_timer("tile APR");

    BenchmarkAPR::tileAPR(tile_dims,test_data.apr, test_data.particles_intensities,apr_tiled,tiled_parts);

    timer.stop_timer();

    std::cout << "Original number particles: " << test_data.apr.total_number_particles() << std::endl;
    std::cout << "Tiled number particles: " << apr_tiled.total_number_particles() << std::endl;

    std::cout << "Original size: " << test_data.apr.org_dims(0) << "x" << test_data.apr.org_dims(1) << "x" << test_data.apr.org_dims(2) << std::endl;
    std::cout << "New size: " << apr_tiled.org_dims(0) << "x" << apr_tiled.org_dims(1) << "x" << apr_tiled.org_dims(2) << std::endl;

    std::cout << "Old CR: " << test_data.apr.computational_ratio() << std::endl;
    std::cout << "Tiled CR: " << apr_tiled.computational_ratio() << std::endl;

    PixelData<uint16_t> level_img;

    APRReconstruction::interp_level(test_data.apr,level_img);

    TiffUtils::saveMeshAsTiff("level_org.tif", level_img);

    APRReconstruction::interp_level(apr_tiled,level_img);

    TiffUtils::saveMeshAsTiff("level_tiled.tif", level_img);

    APRReconstruction::interp_img(test_data.apr,level_img,test_data.particles_intensities);

    TiffUtils::saveMeshAsTiff("parts.tif", level_img);

    APRReconstruction::interp_img(apr_tiled,level_img,tiled_parts);

    TiffUtils::saveMeshAsTiff("parts_tiled.tif", level_img);

    APRFile aprFile;
    aprFile.set_write_linear_flag(true);
    aprFile.open("tiledAPR.apr","WRITE");
    aprFile.write_apr(apr_tiled);
    aprFile.write_particles(apr_tiled,"parts",tiled_parts);
    aprFile.close();

    return true;
}

bool bench_particle_structures(TestData& test_data) {
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///

    bool success = true;

    auto it = test_data.apr.iterator();

    ParticleData<uint16_t> parts;
    parts.init(it.total_number_particles());

    PixelData<uint16_t> test_img;

    test_img.init(it.org_dims(0), it.org_dims(1), it.org_dims(2));

    float CR = test_img.mesh.size() / (1.0f * it.total_number_particles());

    std::cout << "CR: " << CR << std::endl;

    APRTimer timer(true);

    unsigned int num_rep = 1000;

    auto lin_it = test_data.apr.iterator();

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
    partCellData.initialize_structure_parts(test_data.apr);

    timer.start_timer("LinearIteration - PartCell - OpenMP");

    for (int r = 0; r < num_rep; ++r) {
        for (unsigned int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
            int z = 0;

#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z) firstprivate(lin_it)
#endif
            for (z = 0; z < lin_it.z_num(level); z++) {
                for (int x = 0; x < lin_it.x_num(level); ++x) {
                    for (auto begin = lin_it.begin(level, z, x); lin_it < lin_it.end();
                         lin_it++) {
                        //need to add the ability to get y, and x,z but as possible should be lazy.
                        auto off = z*lin_it.x_num(level) + x;
                        auto indx = lin_it - begin;
                        partCellData.data[level][off][indx] += 1;
                    }
                }
            }
        }
    }


    timer.stop_timer();

    return success;

}


bool bench_iteration(TestData& test_data){
    ///
    /// Tests the pipeline, comparing the results with existing results
    ///


    std::vector<int> tile_dims = {4,3,2};

    APR apr_tiled;
    ParticleData<uint16_t> parts;

    APRTimer timer(true);

    timer.start_timer("tile APR");

    BenchmarkAPR::tileAPR(tile_dims,test_data.apr, test_data.particles_intensities,apr_tiled,parts);

    bool success = true;

    auto it = apr_tiled.random_iterator();

    PixelData<uint16_t> test_img;

    test_img.init(it.org_dims(0),it.org_dims(1),it.org_dims(2));

    float CR = test_img.mesh.size()/(1.0f*it.total_number_particles());

    std::cout << "CR: " << CR << std::endl;

    unsigned int num_rep = 1;

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

    int z = 0;

    for (int r = 0; r < num_rep; ++r) {
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

    //y iteration here.
    std::vector<uint16_t> y_vec;
    std::vector<uint64_t> xz_end_vec;
    std::vector<uint64_t> level_end_vec;
    std::vector<uint64_t> level_xz_vec;
    uint64_t counter = 0;
    uint64_t counter_xz = 1;

    level_end_vec.resize(it.level_max() + 1);
    level_xz_vec.resize(it.level_max() + 1);

    xz_end_vec.push_back(counter); // adding padding by one to allow the -1 syntax without checking.

    for (unsigned int level = 0; level <= it.level_max(); ++level) {
        int z = 0;
        int x = 0;

        for (z = 0; z < it.z_num(level); z++) {
            for (x = 0; x < it.x_num(level); ++x) {

                for (it.set_new_lzx(level, z, x); it < it.end();
                     it++) {
                    y_vec.push_back(it.y());
                    counter++;
                }

                xz_end_vec.push_back(counter);
                counter_xz++;
            }
        }

        level_end_vec[level] = counter;
        level_xz_vec[level] = counter_xz;
    }


    timer.start_timer("APR no access - Serial");

    for (int r = 0; r < num_rep; ++r) {

        for (int i = 0; i < it.total_number_particles(); ++i) {
            parts[i] = y_vec[i];
        }

    }
    timer.stop_timer();


    timer.start_timer("APR no access - OpenMP");

    for (int r = 0; r < num_rep; ++r) {
        int i = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for private(i)
#endif
        for (i = 0; i < it.total_number_particles(); ++i) {
            parts[i] = y_vec[i];
        }

    }
    timer.stop_timer();


    timer.start_timer("APR Iteration NEW - OpenMP");

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

    uint64_t counter_test = 0;

    for (unsigned int level = lin_it.level_min(); level <= lin_it.level_max(); ++level) {
        int z = 0;

        for (z = 0; z < lin_it.z_num(level); z++) {
            for (int x = 0; x < lin_it.x_num(level); ++x) {
                for (lin_it.begin(level, z, x); lin_it < lin_it.end();
                     lin_it++) {
                    parts[lin_it] = (uint16_t)(parts[lin_it] + 1);
                    counter_test++;
                }
            }
        }
    }


    timer.stop_timer();

    std::cout << counter_test << std::endl;
    std::cout << test_data.apr.total_number_particles()*num_rep << std::endl;

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

    //read in the command line options into the parameters file
    apr.parameters.Ip_th = 0;
    apr.parameters.rel_error = rel_error;
    apr.parameters.lambda = 0;
    apr.parameters.mask_file = "";
    apr.parameters.min_signal = -1;

    apr.parameters.sigma_th_max = 50;
    apr.parameters.sigma_th = 100;

    apr.parameters.SNR_min = -1;

    apr.parameters.auto_parameters = false;

    apr.parameters.output_steps = true;

    //where things are
    apr.parameters.input_image_name = test_data.filename;
    apr.parameters.input_dir = "";
    apr.parameters.name = test_data.output_name;
    apr.parameters.output_dir = test_data.output_dir;

    //Gets the APR
    APRConverter<uint16_t> aprConverter;
    aprConverter.par = apr.parameters;

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

void CreateGTSmallTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_small/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_small/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_small/";
}

void CreateGTSmall2DTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_2D/";
}

void CreateGTSmall1DTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_1D/";
}




void CreateSmallSphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_apr.h5";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_type.tif";
    test_data.img_type = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_pc.tif";
    test_data.img_pc = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_x.tif";
    test_data.img_x = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_120/sphere_y.tif";
    test_data.img_y = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_120/sphere_z.tif";
    test_data.img_z = TiffUtils::getMesh<uint16_t>(file_name,false);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_120/sphere_original.tif";
    test_data.output_name = "sphere_small";
}

void Create210SphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_apr.h5";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name,false);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_210/sphere_original.tif";
    test_data.output_name = "sphere_210";
}



TEST_F(Create210SphereTest, BENCH_ITERATION) {

    ASSERT_TRUE(bench_iteration(test_data));

}

TEST_F(Create210SphereTest, BENCH_STRUCTURES) {

    //ASSERT_TRUE(bench_particle_structures(test_data));

}

TEST_F(CreateSmallSphereTest, BENCH_TILE) {

    //ASSERT_TRUE(test_tiling(test_data));

}


TEST_F(Create210SphereTest, BENCH_TILE) {

    ASSERT_TRUE(test_tiling(test_data));

}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}

