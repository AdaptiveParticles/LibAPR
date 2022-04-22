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

#include "numerics/PixelNumericsGPU.hpp"
#include "numerics/APRNumericsGPU.hpp"


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


void CreateSmallSphereTest::SetUp(){


    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
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
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR3::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_3.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR5::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_5.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR10::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_10.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR15::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_15.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR20::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_20.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR30::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_30.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR54::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_54.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR124::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_124.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}

void CreateCR1000::SetUp(){


    std::string file_name = get_source_directory_apr() + "../benchmarks/files/cr_1000.apr";
    test_data.apr_filename = file_name;

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.read_apr(test_data.apr);
    aprFile.read_particles(test_data.apr,"particle_intensities",test_data.particles_intensities);
    aprFile.close();
}


#ifdef APR_USE_CUDA


bool test_down_sample_gpu(TestDataGPU& test_data){

    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuDataTree.init_gpu();
    gpuData.init_gpu(gpuDataTree);

    VectorData<float> tree_data;

    downsample_avg(gpuData, gpuDataTree, test_data.particles_intensities.data, tree_data);



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
//                        std::cout << rows << std::endl;
//                        std::cout << "x: " << x << " z: " << z << std::endl;
                    }
                }
            }
        }
    }

    std::cout << "rows filled: " << rows_filled << " / " << rows << std::endl;
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

bool test_gpu_conv_333_alt(TestDataGPU& test_data, bool use_stencil_downsample){
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

    isotropic_convolve_333_alt(access, tree_access, test_data.particles_intensities.data, output, stencil, tree_data, use_stencil_downsample, false);

    std::vector<PixelData<double>> stencil_vec;
    int nstencils = use_stencil_downsample ? access.level_max() - access.level_min() : 1;
    stencil_vec.resize(nstencils);

    stencil_vec[0].init(3, 3, 3);
    std::copy(stencil.begin(), stencil.end(), stencil_vec[0].mesh.begin());

    if(use_stencil_downsample) {
        int c = 1;
        PixelData<double> stencil_ds;
        for (int level = access.level_max() - 1; level > access.level_min(); --level) {
            APRStencil::downsample_stencil(stencil_vec[0], stencil_ds, access.level_max() - level, false);
            stencil_vec[c].init(stencil_ds);
            stencil_vec[c].copyFromMesh(stencil_ds);
            c++;
        }
    }

    ParticleData<double> output_gt;
    APRFilter::create_test_particles_equiv(test_data.apr, stencil_vec, test_data.particles_intensities, output_gt, false);

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

TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateSmallSphereTest, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR1, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR3, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR5, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR10, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR15, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR20, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR30, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR54, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR124, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}

TEST_F(CreateCR1000, TEST_GPU_CONV_333_ALT) {
    std::cout << "with stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, true));
    std::cout << "Without stencil downsample:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333_alt(test_data, false));
}


bool test_gpu_conv_333(TestDataGPU& test_data, bool reflective_bc, bool use_stencil_downsample){
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

    isotropic_convolve_333(access, tree_access, test_data.particles_intensities.data, output, stencil,
                        tree_data, reflective_bc, use_stencil_downsample, false);

    std::vector<PixelData<double>> stencil_vec;
    int nstencils = use_stencil_downsample ? access.level_max() - access.level_min() : 1;
    stencil_vec.resize(nstencils);

    stencil_vec[0].init(3, 3, 3);
    std::copy(stencil.begin(), stencil.end(), stencil_vec[0].mesh.begin());

    if(use_stencil_downsample) {
        int c = 1;
        PixelData<double> stencil_ds;
        for (int level = access.level_max() - 1; level > access.level_min(); --level) {
            APRStencil::downsample_stencil(stencil_vec[0], stencil_ds, access.level_max() - level, false);
            stencil_vec[c].init(stencil_ds);
            stencil_vec[c].copyFromMesh(stencil_ds);
            c++;
        }
    }

    ParticleData<double> output_gt;
    APRFilter::create_test_particles_equiv(test_data.apr, stencil_vec, test_data.particles_intensities, output_gt, reflective_bc);

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

TEST_F(CreateSmallSphereTest, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR1, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR3, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR5, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR10, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR15, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR20, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR30, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR54, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR124, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}

TEST_F(CreateCR1000, TEST_GPU_CONV_333) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_333(test_data, true, true));
}


bool test_gpu_conv_555_alt(TestDataGPU& test_data, bool use_stencil_downsample) {
    auto access = test_data.apr.gpuAPRHelper();
    auto tree_access = test_data.apr.gpuTreeHelper();

    access.init_gpu();
    tree_access.init_gpu();

    VectorData<double> tree_data;
    VectorData<double> output;
    VectorData<double> stencil;
    stencil.resize(125);

    double sum = 62.0 * 125;
    for(int i = 0; i < 125; ++i) {
        stencil[i] = ((double) i) / sum;
    }

    isotropic_convolve_555_alt(access, tree_access, test_data.particles_intensities.data, output, stencil, tree_data, use_stencil_downsample, false);

    std::vector<PixelData<double>> stencil_vec;
    int nstencils = use_stencil_downsample ? access.level_max()-access.level_min() : 1;
    stencil_vec.resize(nstencils);
    stencil_vec[0].init(5, 5, 5);
    std::copy(stencil.begin(), stencil.end(), stencil_vec[0].mesh.begin());

    if(use_stencil_downsample){
        int c = 1;
        PixelData<double> stencil_ds;
        for (int level = access.level_max() - 1; level > access.level_min(); --level) {
            APRStencil::downsample_stencil(stencil_vec[0], stencil_ds, access.level_max() - level, false);
            stencil_vec[c].init(stencil_ds);
            stencil_vec[c].copyFromMesh(stencil_ds);
            c++;
        }
    }

    ParticleData<double> output_gt;
    APRFilter::create_test_particles_equiv(test_data.apr, stencil_vec, test_data.particles_intensities, output_gt, false);

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


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateSmallSphereTest, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR1, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR3, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR5, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR10, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR15, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR20, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR30, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR54, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR124, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}

TEST_F(CreateCR1000, TEST_GPU_CONV_555_ALT) {
    std::cout << "With downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, true));
    std::cout << "Without downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555_alt(test_data, false));
}


bool test_gpu_conv_555(TestDataGPU& test_data, bool reflective_bc, bool use_stencil_downsample) {
    auto access = test_data.apr.gpuAPRHelper();
    auto tree_access = test_data.apr.gpuTreeHelper();

    access.init_gpu();
    tree_access.init_gpu();

    VectorData<float> tree_data;
    VectorData<float> output;
    VectorData<float> stencil;

    stencil.resize(125);
    float sum = 62.0f * 125.0f;
    for(int i = 0; i < 125; ++i) {
        stencil[i] = ((float) i) / sum;
    }

    isotropic_convolve_555(access, tree_access, test_data.particles_intensities.data, output, stencil,
                        tree_data, reflective_bc, use_stencil_downsample, false);

    std::vector<PixelData<float>> stencil_vec;
    int nstencils = use_stencil_downsample ? access.level_max()-access.level_min() : 1;
    stencil_vec.resize(nstencils);
    stencil_vec[0].init(5, 5, 5);
    std::copy(stencil.begin(), stencil.end(), stencil_vec[0].mesh.begin());

    if(use_stencil_downsample){
        int c = 1;
        PixelData<double> stencil_ds;
        for (int level = access.level_max() - 1; level > access.level_min(); --level) {
            APRStencil::downsample_stencil(stencil_vec[0], stencil_ds, access.level_max() - level, false);
            stencil_vec[c].init(stencil_ds);
            stencil_vec[c].copyFromMesh(stencil_ds);
            c++;
        }
    }

    ParticleData<float> output_gt;
    APRFilter::create_test_particles_equiv(test_data.apr, stencil_vec, test_data.particles_intensities, output_gt, reflective_bc);

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


TEST_F(CreateSmallSphereTest, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR1, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR3, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR5, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR10, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR15, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR20, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR30, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR54, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR124, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}

TEST_F(CreateCR1000, TEST_GPU_CONV_555) {
    std::cout << "zero pad:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, false));
    std::cout << "zero pad with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, false, true));
    std::cout << "reflective boundary:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, false));
    std::cout << "reflective boundary with downsample stencil:" << std::endl;
    ASSERT_TRUE(test_gpu_conv_555(test_data, true, true));
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_PIXEL_333) {

    PixelData<float> output;
    PixelData<float> stencil(3, 3, 3);

    float sum = 13.0 * 27;
    for(int i = 0; i < 27; ++i) {
        stencil.mesh[i] = ((float) i) / sum;
    }

    convolve_pixel_333(test_data.img_original, output, stencil, false);

    PixelData<float> output_gt;

    APRFilter::convolve_pixel(test_data.img_original, output_gt, stencil);

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

    convolve_pixel_555(test_data.img_original, output, stencil, false);

    PixelData<float> output_gt;

    APRFilter::convolve_pixel(test_data.img_original, output_gt, stencil);

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

    VectorData<float> output;
    PixelData<float> psf(5, 5, 5, 1.0f/125.0f);

    richardson_lucy(access, tree_access, test_data.particles_intensities.data, output, psf, 10, true, true);

    return true;
}

TEST_F(CreateSmallSphereTest, TEST_LR) {
    ASSERT_TRUE( run_richardson_lucy(test_data) );
}


bool run_richardson_lucy_pixel(TestDataGPU& test_data) {

    PixelData<float> output;
    PixelData<float> psf(5, 5, 5, 1.0f/125.0f);

    richardson_lucy_pixel(test_data.img_original, output, psf, 10);

    return true;
}

TEST_F(CreateSmallSphereTest, TEST_LR_PIXEL) {
    ASSERT_TRUE( run_richardson_lucy_pixel(test_data) );
}


bool test_ne_rows_cuda(TestDataGPU& test_data, int blockSize) {

    auto tree_access = test_data.apr.gpuTreeHelper();
    auto access = test_data.apr.gpuAPRHelper();

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    cudaDeviceSynchronize();

    VectorData<int> ne_count_cuda;
    VectorData<int> ne_rows_cuda;
    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_cuda_device;

    compute_ne_rows_cuda<16, 32>(access, ne_count_cuda, ne_rows_cuda_device, blockSize);
    cudaDeviceSynchronize();

    /// get the data back to CPU for comparison
    ne_rows_cuda.resize(ne_count_cuda.back());
    cudaMemcpy(ne_rows_cuda.data(), ne_rows_cuda_device.get(), ne_count_cuda.back() * sizeof(int), cudaMemcpyDeviceToHost);

    VectorData<int> ne_count;
    VectorData<int> ne_rows;

    compute_ne_rows(access, ne_count, ne_rows, blockSize);

    bool count_success = true;
    std::cout << "Comparing row counts..." << std::endl;
    for(int level = access.level_max(); level > access.level_min(); --level) {
        int cuda_val = ne_count_cuda[level+1] - ne_count_cuda[level];
        int cpu_val = ne_count[level+1] - ne_count[level];

        if(cuda_val != cpu_val) {
            std::cout << "Error at level " << level << " expected " << cpu_val << " but got " << cuda_val << std::endl;
            count_success = false;
        }
    }

    if(count_success) {
        std::cout << "non-empty row counts OK!\n" << std::endl;
    } else {
        std::cerr << "non-empty row counts failed!\n" << std::endl;
    }

    int successes = 0;

    std::cout << "Comparing row entries..." << std::endl;
    bool row_success = true;

    for(int level = access.level_min(); level <= access.level_max(); ++level) {

        for (int i = ne_count[level]; i < ne_count[level + 1]; ++i) {
            int row = ne_rows[i];
            int occurences = 0;

            for (int j = ne_count_cuda[level]; j < ne_count_cuda[level + 1]; ++j) {
                if (ne_rows_cuda[j] == row) {
                    occurences++;
                }
            }

            if(occurences == 1) {
                successes++;
            } else if (occurences < 1) {
                std::cerr << "level " << level << ": ne_rows_cuda is missing row " << row << std::endl;
                row_success = false;
            } else { // occurences > 1
                std::cerr << "level " << level << ": ne_rows_cuda has " << occurences << " duplicates of row " << row << std::endl;
                row_success = false;
            }
        }
    }

    if(row_success) {
        std::cout << "non-empty rows OK!\n" << std::endl;
    } else {
        std::cerr << "non-empty rows failed!\n" << std::endl;
    }

    int failures = ne_count[access.level_max() + 1]-successes;
    std::cout << "successes: " << successes << " failures: " << failures << std::endl;

    return (row_success && count_success);
}


TEST_F(CreateSmallSphereTest, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));
}

TEST_F(CreatDiffDimsSphereTest, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));
}

TEST_F(CreateCR1, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));
}

TEST_F(CreateCR3, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));
}

TEST_F(CreateCR5, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));}

TEST_F(CreateCR10, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));}

TEST_F(CreateCR15, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));}

TEST_F(CreateCR20, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));}

TEST_F(CreateCR30, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));}

TEST_F(CreateCR54, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));}

TEST_F(CreateCR124, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));}

TEST_F(CreateCR1000, TEST_NE_ROWS_CUDA) {
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 2));
    ASSERT_TRUE(test_ne_rows_cuda(test_data, 4));
}

bool test_ne_rows_tree_cuda(TestDataGPU& test_data) {
    auto tree_access = test_data.apr.gpuTreeHelper();
    auto access = test_data.apr.gpuAPRHelper();

    tree_access.init_gpu();
    access.init_gpu(tree_access);

    cudaDeviceSynchronize();

    VectorData<int> ne_count_cuda;
    VectorData<int> ne_rows_cuda;

    ScopedCudaMemHandler<int*, JUST_ALLOC> ne_rows_cuda_device;
    compute_ne_rows_tree_cuda<16, 32>(tree_access, ne_count_cuda, ne_rows_cuda_device);

    /// get the data back to CPU for comparison
    cudaDeviceSynchronize();
    ne_rows_cuda.resize(ne_count_cuda.back());
    cudaMemcpy(ne_rows_cuda.data(), ne_rows_cuda_device.get(), ne_count_cuda.back() * sizeof(int), cudaMemcpyDeviceToHost);

    VectorData<int> ne_count;
    VectorData<int> ne_rows;

    compute_ne_rows_tree(tree_access, ne_count, ne_rows);

    bool count_success = true;
    std::cout << "Comparing row counts..." << std::endl;
    for(int level = access.level_max(); level > access.level_min(); --level) {
        int cuda_val = ne_count_cuda[level+1] - ne_count_cuda[level];
        int cpu_val = ne_count[level+1] - ne_count[level];

        if(cuda_val != cpu_val) {
            std::cout << "Error at level " << level << " expected " << cpu_val << " but got " << cuda_val << std::endl;
            count_success = false;
        }
    }

    if(count_success) {
        std::cout << "non-empty row counts OK!\n" << std::endl;
    } else {
        std::cerr << "non-empty row counts failed!\n" << std::endl;
    }

    int successes = 0;

    std::cout << "Comparing row entries..." << std::endl;
    bool row_success = true;

    for(int level = access.level_min(); level <= access.level_max(); ++level) {

        for (int i = ne_count[level]; i < ne_count[level + 1]; ++i) {
            int row = ne_rows[i];
            int occurences = 0;

            for (int j = ne_count_cuda[level]; j < ne_count_cuda[level + 1]; ++j) {
                if (ne_rows_cuda[j] == row) {
                    occurences++;
                }
            }

            if(occurences == 1) {
                successes++;
            } else if (occurences < 1) {
                std::cerr << "level " << level << ": ne_rows_cuda is missing row " << row << std::endl;
                row_success = false;
            } else { // occurences > 1
                std::cerr << "level " << level << ": ne_rows_cuda has " << occurences << " duplicates of row " << row << std::endl;
                row_success = false;
            }
        }
    }

    if(row_success) {
        std::cout << "non-empty rows OK!\n" << std::endl;
    } else {
        std::cerr << "non-empty rows failed!\n" << std::endl;
    }

    int failures = ne_count[access.level_max() + 1]-successes;
    std::cout << "successes: " << successes << " failures: " << failures << std::endl;

    return (row_success && count_success);
}


TEST_F(CreateSmallSphereTest, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreatDiffDimsSphereTest, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR1, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR3, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR5, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR10, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR15, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR20, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR30, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR54, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR124, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}

TEST_F(CreateCR1000, TEST_NE_ROWS_TREE_CUDA) {
    ASSERT_TRUE( test_ne_rows_tree_cuda(test_data));
}



TEST_F(CreateSmallSphereTest, CHECK_DOWNSAMPLE_STENCIL) {

    PixelData<float> stencil_pd(5, 5, 5);
    VectorData<float> stencil_vd;
    stencil_vd.resize(125);

    float sum = 62.0f * 125.0f;
    for(int i = 0; i < 125; ++i) {
        stencil_pd.mesh[i] = ((float) i) / sum;
        stencil_vd[i] = ((float) i) / sum;
    }

    VectorData<float> stencil_vec_vd;
    VectorData<float> stencil_vec_pd;
    std::vector<PixelData<float>> pd_vec;

    int nlevels = 7;

    APRStencil::get_downsampled_stencils(stencil_pd, stencil_vec_pd, nlevels, false);
    APRStencil::get_downsampled_stencils(stencil_pd, pd_vec, nlevels, false);
    APRStencil::get_downsampled_stencils(stencil_vd, stencil_vec_vd, nlevels, false);

    // compare outputs for PixelData and VectorData inputs
    bool success = true;
    ASSERT_EQ(stencil_vec_vd.size(), stencil_vec_pd.size());

    std::cout << "comparing downsampled stencils for VectorData and PixelData inputs" << std::endl;

    for(size_t i = 0; i < stencil_vec_pd.size(); ++i) {
        if( std::abs( stencil_vec_pd[i] - stencil_vec_vd[i] ) > 1e-5 ) {
            std::cout << "stencil_vec_vd = " << stencil_vec_vd[i] << " stencil_vec_pd = " << stencil_vec_pd[i] << " at index " << i << std::endl;
            success = false;
        }
    }

    if(success) {
        std::cout << "OK!" << std::endl;
    }

    std::cout << "comparing downsampeld stencils for VectorData and std::vector<PixelData> output" << std::endl;
    success = true;
    int c = 0;
    for(size_t dlvl = 0; dlvl < pd_vec.size(); ++dlvl) {
        for(size_t i = 0; i < pd_vec[dlvl].mesh.size(); ++i) {
            if( std::abs( pd_vec[dlvl].mesh[i] - stencil_vec_pd[c] ) > 1e-5 ) {
                std::cout << "pd_vec = " << pd_vec[dlvl].mesh[i] << " stencil_vec_pd = " << stencil_vec_pd[c] <<
                            " at dlvl = " << dlvl << " and i = " << i << std::endl;
                success = false;
            }
            c++;
        }
    }

    ASSERT_EQ(c, stencil_vec_pd.size());
    if(success) {
        std::cout << "OK!" << std::endl;
    }

}

#endif

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
