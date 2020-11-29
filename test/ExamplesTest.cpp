//
// Created by bevan on 29/11/2020.
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

#ifdef APR_DENOISE
#include "../examples/Example_denoise.hpp"
#endif

struct TestData{

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

class CreateBigBigData : public CreateAPRTest
{
public:
    void SetUp() override;
};


class CreatDiffDimsSphereTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

class Create210SphereTestAPROnly : public CreateAPRTest
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

class CreateGTSmall2DTestProperties : public CreateAPRTest
{
public:
    void SetUp() override;
};


class CreateGTSmall1DTestProperties : public CreateAPRTest
{
public:
    void SetUp() override;
};



class CreateGTSmall2DTestAPR : public CreateAPRTest
{
public:
    void SetUp() override;
};



class CreateGTSmall1DTest : public CreateAPRTest
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

void CreateGTSmallTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_small/original.tif";

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_small/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_small/";
}

void CreateGTSmall2DTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_2D/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_2D/";
}

void CreateGTSmall1DTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_1D/original.tif";
    test_data.output_name = "sphere_gt";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_1D/";
}


void CreateSmallSphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere.apr";
    test_data.apr_filename = file_name;

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_120/sphere_original.tif";
    test_data.output_name = "sphere_small";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_120/";
}

void CreatDiffDimsSphereTest::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere.apr";
    test_data.apr_filename = file_name;

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_diff_dims/sphere_original.tif";
    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_diff_dims/";
    test_data.output_name = "sphere_210";
}

void Create210SphereTestAPROnly::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_210/sphere_apr.h5";
    test_data.apr_filename = file_name;

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_210/sphere_original.tif";
    test_data.output_name = "sphere_210";
}


void CreateGTSmall2DTestProperties::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D.apr";
    test_data.apr_filename = file_name;

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D_original.tif";
    test_data.output_name = "sphere_2D";
    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_2D/";
}

void CreateGTSmall1DTestProperties::SetUp(){

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D.apr";
    test_data.apr_filename = file_name;

    test_data.filename = get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D_original.tif";
    test_data.output_name = "sphere_1D";

    test_data.output_dir = get_source_directory_apr() + "files/Apr/sphere_1D/";
}

#ifdef APR_DENOISE

bool run_denoise_example(TestData& testData){

    cmdLineOptionsDenoise options;

    options.input = testData.apr_filename;

    return denoise_example(options);

}


//1D
TEST_F(CreateGTSmall1DTestProperties, EXAMPLE_DENOISE) {

    ASSERT_TRUE(run_denoise_example(test_data));

}

//2D
TEST_F(CreateGTSmall2DTestProperties, EXAMPLE_DENOISE) {

    ASSERT_TRUE(run_denoise_example(test_data));

}

//3D
TEST_F(CreateSmallSphereTest, EXAMPLE_DENOISE) {

    ASSERT_TRUE(run_denoise_example(test_data));

}

//3D
TEST_F(CreatDiffDimsSphereTest, EXAMPLE_DENOISE) {

    ASSERT_TRUE(run_denoise_example(test_data));

}

#endif

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
