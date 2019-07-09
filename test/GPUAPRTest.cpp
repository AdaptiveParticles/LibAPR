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



TEST_F(CreatDiffDimsSphereTest, APR_TEST) {

    test_data.apr.init_gpu();

    ASSERT_TRUE(run_simple_test());

}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
