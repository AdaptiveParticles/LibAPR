//
// Created by cheesema on 21.01.18.
//


#include <gtest/gtest.h>
#include "numerics/APRFilter.hpp"
#include "data_structures/APR/APR.hpp"
#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/APRConverter.hpp"
#include <utility>
#include <cmath>
#include "TestTools.hpp"
#include "numerics/APRTreeNumerics.hpp"
#include "io/APRWriter.hpp"

#include "data_structures/APR/particles/LazyData.hpp"

#include "io/APRFile.hpp"

#ifndef APR_USE_CUDA

bool run_simple_test(){
        return true;
};

#else
    #include "GPUAPR.hpp"
    #include "numerics/APRDownsampleGPU.hpp"
#endif

struct TestDataGPU{

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

class CreateGPUAPRTest : public ::testing::Test {
public:

    TestDataGPU test_data;

protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

};

class CreateSmallSphereTest : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};


class CreatDiffDimsSphereTest : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};





std::string get_source_directory_apr(){
    // returns path to the directory where utils.cpp is stored

    std::string tests_directory = std::string(__FILE__);
    tests_directory = tests_directory.substr(0, tests_directory.find_last_of("\\/") + 1);

    return tests_directory;
}



void CreateSmallSphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name,false);
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

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_120/";
}

void CreatDiffDimsSphereTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();

    file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_level.tif";
    test_data.img_level = TiffUtils::getMesh<uint16_t>(file_name,false);

    file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_original.tif";
    test_data.img_original = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_pc.tif";
    test_data.img_pc = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_x.tif";
    test_data.img_x = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_y.tif";
    test_data.img_y = TiffUtils::getMesh<uint16_t>(file_name,false);
    file_name =  get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_z.tif";
    test_data.img_z = TiffUtils::getMesh<uint16_t>(file_name,false);

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_original.tif";
    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_diff_dims/";
    test_data.output_name = "sphere_210";
}


TEST_F(CreatDiffDimsSphereTest, APR_ACCESS_TEST) {

    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuData.init_gpu();
    gpuDataTree.init_gpu();

    gpuData.copy2Host();

    auto apr_it = test_data.apr.iterator();

    uint64_t counter = 0;

    for(unsigned int level = apr_it.level_max(); level > apr_it.level_min(); --level) {
        for(size_t z = 0; z < apr_it.z_num(level); ++z) {
            for(size_t x = 0; x < apr_it.x_num(level); ++x) {
                for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
                    counter++;
                }
            }
        }
    }

    ASSERT_TRUE(counter == test_data.apr.total_number_particles());

}


TEST_F(CreatDiffDimsSphereTest, APR_ACCESS_ROUNDTRIP_TEST) {

    auto gpuData = test_data.apr.gpuAPRHelper();
    //auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuData.init_gpu();
    //gpuDataTree.init_gpu();

    std::vector<uint64_t> y_vec;
    std::vector<uint64_t> xz_end_vec;
    std::vector<uint64_t> level_xz_vec;

    check_access_vectors(gpuData, y_vec, xz_end_vec, level_xz_vec);

    uint64_t success_y = 0, success_xz = 0, success_lvl = 0;

    for(size_t i = 0; i < gpuData.linearAccess->y_vec.size(); ++i) {
        if(y_vec[i] == gpuData.linearAccess->y_vec[i]) {
            success_y++;
        }
    }

    for(size_t i = 0; i < gpuData.linearAccess->xz_end_vec.size(); ++i) {
        if(xz_end_vec[i] == gpuData.linearAccess->xz_end_vec[i]) {
            success_xz++;
        }
    }

    for(size_t i = 0; i < gpuData.linearAccess->level_xz_vec.size(); ++i) {
        if(level_xz_vec[i] == gpuData.linearAccess->level_xz_vec[i]) {
            success_lvl++;
        }
    }

    ASSERT_EQ(success_lvl, level_xz_vec.size());
    ASSERT_EQ(success_xz, xz_end_vec.size());
    ASSERT_EQ(success_y, y_vec.size());
}


TEST_F(CreatDiffDimsSphereTest, SIMPLE_DATA_TEST) {

    uint64_t size = 1000;
    std::vector<uint64_t> temp;

    run_simple_test(temp, size);

    uint64_t successes = 0;

    for(size_t idx = 0; idx < size; ++idx) {
        if(temp[idx] == idx) {
            successes++;
        }
    }

    ASSERT_EQ(successes, size);
}


TEST_F(CreatDiffDimsSphereTest, APR_TEST) {

    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    auto apr_it = test_data.apr.iterator();

    gpuData.init_gpu();
    gpuDataTree.init_gpu();

    std::vector<uint64_t> spatial_info_gpu;
    spatial_info_gpu.resize(apr_it.total_number_particles());

    compute_spatial_info_gpu(gpuData, spatial_info_gpu);

    std::vector<uint64_t> spatial_info_cpu;
    spatial_info_cpu.resize(apr_it.total_number_particles());

    for(unsigned int level = apr_it.level_max(); level > apr_it.level_min(); --level) {
        for(size_t z = 0; z < apr_it.z_num(level); ++z) {
            for(size_t x = 0; x < apr_it.x_num(level); ++x) {
                for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {
                    spatial_info_cpu[apr_it] = level + z + x + apr_it.y();
                }
            }
        }
    }

    uint64_t successes = 0;
    for(int i = 0; i < spatial_info_cpu.size(); ++i) {
        if(spatial_info_cpu[i] == spatial_info_gpu[i]) {
            successes++;
        }
    }

    ASSERT_EQ(successes, apr_it.total_number_particles());

}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_DOWNSAMPLE) {

    auto apr_it = test_data.apr.iterator();
    std::vector<uint16_t> parts;
    parts.resize(apr_it.total_number_particles());

    for(size_t i = 0; i < parts.size(); ++i) {
        parts[i] = test_data.particles_intensities[i];
    }

    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuData.init_gpu();
    gpuDataTree.init_gpu();

    std::vector<uint16_t> tree_data;

    downsample_avg(gpuData, gpuDataTree, parts, tree_data);

    ParticleData<float> tree_data_cpu;
    APRTreeNumerics::fill_tree_mean(test_data.apr, test_data.particles_intensities, tree_data_cpu);

    size_t successes = 0;
    for(size_t i = 0; i < tree_data.size(); ++i){
        if(abs(tree_data[i] - tree_data_cpu[i]) < 1e-2) {
            successes++;
        }
    }

    ASSERT_EQ(successes, tree_data.size());
}


int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
