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

#ifdef APR_USE_CUDA
#include "numerics/APRDownsampleGPU.hpp"
#include "numerics/APRIsoConvGPU.hpp"

#define DEBUGCUDA 1

#define error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
#ifdef DEBUGCUDA
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
#endif
}
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


class CreateCR1 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR3 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR5 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR10 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR15 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR20 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR30 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR54 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR124 : public CreateGPUAPRTest
{
public:
    void SetUp() override;
};

class CreateCR1000 : public CreateGPUAPRTest
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


void CreateCR1::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_1.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR3::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_3.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR5::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_5.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR10::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_10.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR15::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_15.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR20::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_20.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR30::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_30.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR54::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_54.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR124::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_124.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR1000::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_1000.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}


#ifdef APR_USE_CUDA


bool test_down_sample_gpu(TestDataGPU& test_data){
    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuData.init_gpu();
    gpuDataTree.init_gpu();

    VectorData<float> tree_data;

    downsample_avg_alt(gpuData, gpuDataTree, test_data.particles_intensities.data, tree_data);



    ParticleData<float> tree_data_cpu;
    APRTreeNumerics::fill_tree_mean(test_data.apr, test_data.particles_intensities, tree_data_cpu);

    size_t successes = 0;

    auto tree_it = test_data.apr.tree_iterator();

    uint64_t counter = 0;
    uint64_t rows = 0;
    uint64_t rows_filled = 0;

    for (int l = tree_it.level_min(); l <= tree_it.level_max(); ++l) {
        for (int z = 0; z < tree_it.z_num(l); ++z) {
            for (int x = 0; x < tree_it.x_num(l); ++x) {

                bool filled = false;

                for (tree_it.begin(l,z,x); tree_it < tree_it.end(); tree_it++) {
                    if(std::abs(tree_data[tree_it] - tree_data_cpu[tree_it]) < 1e-2) {

                        successes++;
                        filled=true;
                    } else {
                        std::cout << "gpu: " << tree_data[tree_it] << " cpu: " << tree_data_cpu[tree_it] << " at part " << tree_it <<
                        " at level " << l << " z: " << z  << " x: " << x << " y: " << tree_it.y() << std::endl;
                    }
                    counter++;
                }

                if(tree_it.begin(l,z,x) != tree_it.end()){
                    rows++;
                    if(filled){
                        rows_filled++;
                    } else {
                        std::cout << rows << std::endl;
                        std::cout << "x: " << x << " z: " << z << std::endl;
                    }
                }
            }
        }
    }

    std::cout << "successes: " << successes << " / " << tree_it.total_number_particles() << std::endl;

    return (successes == counter);
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateSmallSphereTest, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR1, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR3, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR5, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR10, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR15, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR20, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR30, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR54, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR124, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

TEST_F(CreateCR1000, TEST_GPU_DOWNSAMPLE) {

    ASSERT_TRUE(test_down_sample_gpu(test_data));
}

bool test_gpu_conv_333(TestDataGPU& test_data, bool use_ne_rows){
    auto access = test_data.apr.gpuAPRHelper();
    auto tree_access = test_data.apr.gpuTreeHelper();

    access.init_gpu();
    //tree_access.init_gpu();

    VectorData<double> tree_data;
    VectorData<double> output;
    VectorData<double> stencil;
    stencil.resize(27);

    float sum = 13.0 * 27;
    for(int i = 0; i < 27; ++i) {
        stencil[i] = ((double) i) / sum;
    }

    if(use_ne_rows){
        isotropic_convolve_333(access, tree_access, test_data.particles_intensities.data, output, stencil, tree_data);
    } else {
        isotropic_convolve_333_alt(access, tree_access, test_data.particles_intensities.data, output, stencil, tree_data);
    }

    std::vector<PixelData<double>> stencils;
    stencils.resize(1);

    stencils[0].init(3, 3, 3);
    std::copy(stencil.begin(), stencil.end(), stencils[0].mesh.begin());

    APRFilter filterfns;
    filterfns.boundary_cond = false; // zero padding

    ParticleData<double> output_gt;
    filterfns.create_test_particles_equiv(test_data.apr, stencils, test_data.particles_intensities, output_gt);

    size_t pass_count = 0;
    size_t total_count = 0;

    auto it = test_data.apr.iterator();

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(std::abs(output[it] - output_gt[it]) < 1e-2) {
                        pass_count++;
                    } else {
                        std::cout << "Expected " << output_gt[it] << " but received " << output[it] <<
                                  " at particle index " << it << " (level, z, x, y) = (" << level << ", " << z << ", " << x << ", " << it.y() << ")" << std::endl;
                    }
                    total_count++;
                }
            }
        }
    }

    std::cout << "passed: " << pass_count << " failed: " << test_data.apr.total_number_particles()-pass_count << std::endl;

    return (pass_count == total_count);
}

TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateSmallSphereTest, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR1, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR3, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR5, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR10, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR15, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR20, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR30, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR54, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR124, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}

TEST_F(CreateCR1000, TEST_GPU_CONV_333) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false));
}


bool test_gpu_conv_555(TestDataGPU& test_data, bool use_ne_rows) {
    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuData.init_gpu();
    gpuDataTree.init_gpu();

    VectorData<double> tree_data;
    VectorData<double> output;
    VectorData<double> stencil;

    stencil.resize(125);
    double sum = 62.0 * 125;
    for(int i = 0; i < 125; ++i) {
        stencil[i] = ((double) i) / sum;
    }

    if(use_ne_rows) {
        isotropic_convolve_555(gpuData, gpuDataTree, test_data.particles_intensities.data, output, stencil, tree_data);
    } else {
        isotropic_convolve_555_alt(gpuData, gpuDataTree, test_data.particles_intensities.data, output, stencil, tree_data);
    }

    std::vector<PixelData<double>> stencils;
    stencils.resize(1);
    stencils[0].init(5, 5, 5);
    std::copy(stencil.begin(), stencil.end(), stencils[0].mesh.begin());

    APRFilter filterfns;
    filterfns.boundary_cond = false; // zero padding

    ParticleData<double> output_gt;
    filterfns.create_test_particles_equiv(test_data.apr, stencils, test_data.particles_intensities, output_gt);

    size_t pass_count = 0;
    size_t total_count = 0;

    auto it = test_data.apr.iterator();

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(std::abs(output[it] - output_gt[it]) < 1e-2) {
                        pass_count++;
                    } else {
                        std::cout << "Expected " << output_gt[it] << " but received " << output[it] <<
                                  " at particle index " << it << " (level, z, x, y) = (" << level << ", " << z << ", " << x << ", " << it.y() << ")" << std::endl;
                    }
                    total_count++;
                }
            }
        }
    }

    std::cout << "passed: " << pass_count << " failed: " << test_data.apr.total_number_particles()-pass_count << std::endl;

    return (pass_count == total_count);
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateSmallSphereTest, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR1, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR3, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR5, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR10, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR15, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR20, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR30, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR54, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR124, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}

TEST_F(CreateCR1000, TEST_GPU_CONV_555) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false));
}


bool test_gpu_conv_555_ds(TestDataGPU& test_data, bool use_ne_rows) {
    auto access = test_data.apr.gpuAPRHelper();
    auto tree_access = test_data.apr.gpuTreeHelper();

    access.init_gpu();
    tree_access.init_gpu();

    VectorData<double> tree_data;
    VectorData<double> output;
    PixelData<double> stencil(5, 5, 5);
//    VectorData<double> stencil;
//    stencil.resize(125);

    double sum = 62.0 * 125;
    for(int i = 0; i < 125; ++i) {
        stencil.mesh[i] = ((double) i) / sum;
    }

    isotropic_convolve_555_ds(access, tree_access, test_data.particles_intensities.data, output, stencil, tree_data, use_ne_rows, false);

    std::vector<PixelData<double>> stencil_vec;
    stencil_vec.resize(access.level_max() - access.level_min());
    stencil_vec[0].init(5, 5, 5);
//    stencil_vec[0].copyFromMesh(stencil);
    std::copy(stencil.mesh.begin(), stencil.mesh.end(), stencil_vec[0].mesh.begin());

    int c = 1;
    PixelData<double> stencil_ds;
    for (int level = access.level_max() - 1; level > access.level_min(); --level) {
        downsample_stencil(stencil, stencil_ds, access.level_max() - level, false);
        stencil_vec[c].init(stencil_ds);
        stencil_vec[c].copyFromMesh(stencil_ds);
        c++;
    }

    APRFilter filterfns;
    filterfns.boundary_cond = false; // zero padding

    ParticleData<double> output_gt;
    filterfns.create_test_particles_equiv(test_data.apr, stencil_vec, test_data.particles_intensities, output_gt);

    size_t pass_count = 0;
    size_t total_count = 0;

    auto it = test_data.apr.iterator();

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(std::abs(output[it] - output_gt[it]) < 1e-2) {
                        pass_count++;
                    } else {
                        std::cout << "Expected " << output_gt[it] << " but received " << output[it] <<
                                  " at particle index " << it << " (level, z, x, y) = (" << level << ", " << z << ", " << x << ", " << it.y() << ")" << std::endl;
                    }
                    total_count++;
                }
            }
        }
    }

    std::cout << "passed: " << pass_count << " failed: " << test_data.apr.total_number_particles()-pass_count << std::endl;

    return (pass_count == total_count);
}

TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateSmallSphereTest, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR1, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR3, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR5, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR10, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR15, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR20, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR30, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR54, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR124, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}

TEST_F(CreateCR1000, TEST_GPU_CONV_555_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_ds(test_data, false));
}


bool test_gpu_conv_333_ds(TestDataGPU& test_data, bool use_ne_rows) {
    auto access = test_data.apr.gpuAPRHelper();
    auto tree_access = test_data.apr.gpuTreeHelper();

    access.init_gpu();
    tree_access.init_gpu();

    VectorData<double> tree_data;
    VectorData<double> output;
    PixelData<double> stencil(3, 3, 3);


    double sum = 13.0 * 27;
    for(int i = 0; i < 27; ++i) {
        stencil.mesh[i] = ((double) i) / sum;
    }

    isotropic_convolve_333_ds(access, tree_access, test_data.particles_intensities.data, output, stencil, tree_data, use_ne_rows, false);

    std::vector<PixelData<double>> stencil_vec;
    stencil_vec.resize(access.level_max() - access.level_min());
    stencil_vec[0].init(3, 3, 3);
    std::copy(stencil.mesh.begin(), stencil.mesh.end(), stencil_vec[0].mesh.begin());
//    stencil_vec[0].copyFromMesh(stencil);

    int c = 1;
    PixelData<double> stencil_ds;
    for (int level = access.level_max() - 1; level > access.level_min(); --level) {
        downsample_stencil(stencil, stencil_ds, access.level_max() - level, false);
        stencil_vec[c].init(stencil_ds);
        stencil_vec[c].copyFromMesh(stencil_ds);
        c++;
    }

    APRFilter filterfns;
    filterfns.boundary_cond = false; // zero padding

    ParticleData<double> output_gt;
    filterfns.create_test_particles_equiv(test_data.apr, stencil_vec, test_data.particles_intensities, output_gt);

    size_t pass_count = 0;
    size_t total_count = 0;

    auto it = test_data.apr.iterator();

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(std::abs(output[it] - output_gt[it]) < 1e-2) {
                        pass_count++;
                    } else {
                        std::cout << "Expected " << output_gt[it] << " but received " << output[it] <<
                                  " at particle index " << it << " (level, z, x, y) = (" << level << ", " << z << ", " << x << ", " << it.y() << ")" << std::endl;
                    }
                    total_count++;
                }
            }
        }
    }

    std::cout << "passed: " << pass_count << " failed: " << test_data.apr.total_number_particles()-pass_count << std::endl;

    return (pass_count == total_count);
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateSmallSphereTest, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR1, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR3, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR5, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR10, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR15, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR20, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR30, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR54, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR124, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}

TEST_F(CreateCR1000, TEST_GPU_CONV_333_DS_STENCIL) {
    std::cout << "With non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, true));
    std::cout << "Without non-empty rows:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_ds(test_data, false));
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_PIXEL_333) {

    PixelData<float> output;
    PixelData<float> stencil(3, 3, 3);

    float sum = 13.0 * 27;
    for(int i = 0; i < 27; ++i) {
        stencil.mesh[i] = ((float) i) / sum;
    }

    convolve_pixel_333(test_data.img_original, output, stencil);

    PixelData<float> output_gt;

    APRFilter filterfns;
    filterfns.convolve_pixel(test_data.img_original, output_gt, stencil);

    auto c_fail = compareMeshes(output_gt, output, /*error threshold*/ 1e-2);

    ASSERT_EQ(c_fail, 0);
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_PIXEL_555) {

    PixelData<float> output;
    PixelData<float> stencil(5, 5, 5);

    double sum = 62.0 * 125;
    for(int i = 0; i < 125; ++i) {
        stencil.mesh[i] = ((double) i) / sum;
    }

    convolve_pixel_555(test_data.img_original, output, stencil);

    PixelData<float> output_gt;

    APRFilter filterfns;
    filterfns.convolve_pixel(test_data.img_original, output_gt, stencil);

    auto c_fail = compareMeshes(output_gt, output, /*error threshold*/ 1e-2);

    ASSERT_EQ(c_fail, 0);
}


TEST_F(CreatDiffDimsSphereTest, TEST_PARTIAL_ACCESS_INIT) {

    auto access = test_data.apr.gpuAPRHelper();
    auto tree_access = test_data.apr.gpuTreeHelper();

    tree_access.init_gpu();

    std::vector<uint16_t> y_vec(test_data.apr.total_number_particles());
    std::copy(access.linearAccess->y_vec.begin(), access.linearAccess->y_vec.end(), y_vec.begin());

    auto it = test_data.apr.iterator();

    auto access_new = test_data.apr.gpuAPRHelper();

    access_new.init_gpu(tree_access);

    access_new.copy2Host();

    size_t c_fail = 0;

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if( (y_vec[it] != access_new.linearAccess->y_vec[it]) ) {
                        c_fail++;
                    }
                }
            }
        }
    }

    ASSERT_EQ(c_fail, 0);
}


bool run_richardson_lucy(TestDataGPU& test_data) {
    auto access = test_data.apr.gpuAPRHelper();
    auto tree_access = test_data.apr.gpuTreeHelper();

    access.init_gpu();
    tree_access.init_gpu();

    VectorData<float> output;
    PixelData<float> psf(5, 5, 5, 1.0f/125.0f);

    VectorData<float> finput;
    finput.resize(access.total_number_particles());
    for(size_t i = 0; i < finput.size(); ++i) {
        finput[i] = test_data.particles_intensities.data[i];
    }

    richardson_lucy(access, tree_access, finput, output, psf, 10, true, true);

    PixelData<float> recon(test_data.img_original, false);

    APRReconstruction::interp_img(test_data.apr, recon, output);

    TiffUtils::saveMeshAsTiff("/home/joel/Documents/output/APRLR.tif", recon);

    return true;
}

TEST_F(CreateSmallSphereTest, TEST_LR) {
    ASSERT_TRUE( run_richardson_lucy(test_data) );
}


TEST_F(CreateSmallSphereTest, TEST_DOWNSAMPLE_STENCIL) {

    const int stencil_size = 5;
    const int nlevels = 8;
    const bool normalize = false;
    const float tol = 1e-4;


    PixelData<float> stencil_pd(stencil_size, stencil_size, stencil_size);

    float n = stencil_pd.mesh.size();
    float sum = n * (n-1) * 0.5f;
    for(size_t i = 0; i < stencil_pd.mesh.size(); ++i) {
        stencil_pd.mesh[i] = ((float) i) / sum;
    }

    VectorData<float> stencil_vd;
    stencil_vd.resize(stencil_pd.mesh.size());

    for(size_t i = 0; i < stencil_pd.mesh.size(); ++i) {
        stencil_vd[i] = stencil_pd.mesh[i];
    }

    VectorData<float> ds_stencil_gt;
    get_downsampled_stencils_bruteforce(stencil_pd, ds_stencil_gt, nlevels, normalize);

    VectorData<float> ds_stencil_pd;
    get_downsampled_stencils(stencil_pd, ds_stencil_pd, nlevels, normalize);

    VectorData<float> ds_stencil_vd;
    get_downsampled_stencils(stencil_vd, ds_stencil_vd, nlevels, normalize);

    bool success_pd = true;

    /// compare for PixelData input
    for(size_t i = 0; i < ds_stencil_gt.size(); ++i) {
        if( std::abs(ds_stencil_pd[i] - ds_stencil_gt[i]) > tol ) {
            std::cout << "get_downsampled_stencils with PixelData input failed for i = " << i << ". Expected " << ds_stencil_gt[i] << " but got " << ds_stencil_pd[i] << std::endl;
            success_pd = false;
        }
    }

    bool success_vd = true;

    /// compare for VectorData input
    for(size_t i = 0; i < ds_stencil_gt.size(); ++i) {
        if( std::abs(ds_stencil_vd[i] - ds_stencil_gt[i]) > tol ) {
            std::cout << "get_downsampled_stencils with PixelData input failed for i = " << i << ". Expected " << ds_stencil_gt[i] << " but got " << ds_stencil_vd[i] << std::endl;
            success_vd = false;
        }
    }


    ASSERT_TRUE(success_pd && success_vd);
}


#endif

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
