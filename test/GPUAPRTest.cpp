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

    #include "GPUAPR.hpp"
    #include "numerics/APRDownsampleGPU.hpp"
    #include "numerics/APRIsoConvGPU.hpp"

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

#ifdef APR_USE_CUDA

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

    std::vector<uint16_t> y_vec;
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


TEST_F(CreatDiffDimsSphereTest, APR_ACCESS_ROUNDTRIP_TEST_TREE) {

    auto gpuData = test_data.apr.gpuTreeHelper();
    gpuData.init_gpu();

    std::vector<uint16_t> y_vec;
    std::vector<uint64_t> xz_end_vec;
    std::vector<uint64_t> level_xz_vec;

    check_access_vectors(gpuData, y_vec, xz_end_vec, level_xz_vec);

    size_t success_y = 0, success_xz = 0, success_lvl = 0;

    for(size_t i = 0; i < gpuData.linearAccess->y_vec.size(); ++i) {
        if(y_vec[i] == gpuData.linearAccess->y_vec[i]) {
            success_y++;
        } else {
            std::cerr << "Received y_vec[" << i << "] = " << y_vec[i] << ", expected " << gpuData.linearAccess->y_vec[i] << std::endl;
        }
    }

    for(size_t i = 0; i < gpuData.linearAccess->xz_end_vec.size(); ++i) {
        if(xz_end_vec[i] == gpuData.linearAccess->xz_end_vec[i]) {
            success_xz++;
        } else {
            std::cerr << "Received xz_end_vec[" << i << "] = " << xz_end_vec[i] << ", expected " << gpuData.linearAccess->xz_end_vec[i] << std::endl;
        }
    }

    for(size_t i = 0; i < gpuData.linearAccess->level_xz_vec.size(); ++i) {
        if(level_xz_vec[i] == gpuData.linearAccess->level_xz_vec[i]) {
            success_lvl++;
        } else {
            std::cerr << "Received level_xz_vec[" << i << "] = " << level_xz_vec[i] << ", expected " << gpuData.linearAccess->level_xz_vec[i] << std::endl;
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

    std::vector<uint16_t> unused_input;
    std::vector<uint16_t> spatial_info_gpu;
    spatial_info_gpu.resize(apr_it.total_number_particles());

    compute_spatial_info_gpu(gpuData, unused_input, spatial_info_gpu);

    std::vector<uint16_t> spatial_info_cpu;
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

    uint64_t c_fail = 0;
    for(int i = 0; i < spatial_info_cpu.size(); ++i) {
        if(spatial_info_cpu[i] != spatial_info_gpu[i]) {
            c_fail++;
            std::cout << "Expected " << spatial_info_cpu[i] << " but received " << spatial_info_gpu[i] << " at index " << i << std::endl;
        }
    }

    ASSERT_EQ(c_fail, 0);

}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_DOWNSAMPLE) {

    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuData.init_gpu();
    gpuDataTree.init_gpu();

    std::vector<float> tree_data;

    downsample_avg(gpuData, gpuDataTree, test_data.particles_intensities.data, tree_data);

    ParticleData<float> tree_data_cpu;
    APRTreeNumerics::fill_tree_mean(test_data.apr, test_data.particles_intensities, tree_data_cpu);

    size_t successes = 0;

#ifdef HAVE_OPENMP
//#pragma omp parallel for reduction(+: successes)
#endif
    for(size_t i = 0; i < tree_data.size(); ++i){
        if(std::abs(tree_data[i] - tree_data_cpu[i]) < 1e-2) {
            successes++;
        } else {
            std::cout << "gpu: " << tree_data[i] << " cpu: " << tree_data_cpu[i] << " at part " << i << std::endl;
        }
    }

    ASSERT_EQ(successes, tree_data.size());
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_333) {

    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuData.init_gpu();
    gpuDataTree.init_gpu();

    std::vector<double> tree_data;
    std::vector<double> output;
    std::vector<double> stencil;
    stencil.resize(27);

    double sum = 13.0 * 27;
    for(int i = 0; i < 27; ++i) {
        stencil[i] = ((double) i) / sum;
    }

    isotropic_convolve_333(gpuData, gpuDataTree, test_data.particles_intensities.data, output, stencil, tree_data);

    std::vector<PixelData<double>> stencils;
    stencils.resize(1);

    stencils[0].init(3, 3, 3);
    std::copy(stencil.begin(), stencil.end(), stencils[0].mesh.begin());

    APRFilter filterfns;
    filterfns.boundary_cond = false; // zero padding

    ParticleData<double> output_gt;
    filterfns.create_test_particles_equiv(test_data.apr, stencils, test_data.particles_intensities, output_gt);

    size_t pass_count = 0;
    bool success = true;

    for(size_t i = 0; i < test_data.apr.total_number_particles(); ++i) {
        if( std::abs(output[i] - output_gt[i]) < 1e-2) {
            pass_count++;
        } else {
            success = false;
            std::cout << "Expected " << output_gt[i] << " but received " << output[i] << " at particle index " << i << std::endl;
        }
    }

    std::cout << "passed: " << pass_count << " failed: " << test_data.apr.total_number_particles()-pass_count << std::endl;
    ASSERT_TRUE(success);
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_555) {

    auto gpuData = test_data.apr.gpuAPRHelper();
    auto gpuDataTree = test_data.apr.gpuTreeHelper();

    gpuData.init_gpu();
    gpuDataTree.init_gpu();

    std::vector<double> tree_data;
    std::vector<double> output;
    std::vector<double> stencil;

    stencil.resize(125);
    double sum = 62.0 * 125;
    for(int i = 0; i < 125; ++i) {
        stencil[i] = ((double) i) / sum;
    }

    isotropic_convolve_555(gpuData, gpuDataTree, test_data.particles_intensities.data, output, stencil, tree_data);

    std::vector<PixelData<double>> stencils;
    stencils.resize(1);
    stencils[0].init(5, 5, 5);
    std::copy(stencil.begin(), stencil.end(), stencils[0].mesh.begin());

    APRFilter filterfns;
    filterfns.boundary_cond = false; // zero padding

    ParticleData<double> output_gt;
    filterfns.create_test_particles_equiv(test_data.apr, stencils, test_data.particles_intensities, output_gt);

    size_t pass_count = 0;
    bool success = true;

    for(size_t i = 0; i < test_data.apr.total_number_particles(); ++i) {
        if( std::abs(output[i] - output_gt[i]) < 1e-2) {
            pass_count++;
        } else {
            success = false;
            std::cout << "Expected " << output_gt[i] << " but received " << output[i] << " at particle index " << i << std::endl;
        }
    }

    std::cout << "passed: " << pass_count << " failed: " << test_data.apr.total_number_particles()-pass_count << std::endl;
    ASSERT_TRUE(success);
}


TEST_F(CreatDiffDimsSphereTest, TEST_GPU_CONV_PIXEL_333) {

    PixelData<double> output;
    PixelData<double> stencil(3, 3, 3);

    double sum = 13.0 * 27;
    for(int i = 0; i < 27; ++i) {
        stencil.mesh[i] = ((double) i) / sum;
    }

    convolve_pixel_333(test_data.img_original, output, stencil);

    PixelData<double> output_gt;

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


    PixelData<float> output_gt;

    APRFilter filterfns;
    filterfns.convolve_pixel(test_data.img_original, output_gt, stencil);

    auto c_fail = compareMeshes(output_gt, output, /*error threshold*/ 1e-2);

    ASSERT_EQ(c_fail, 0);
}

#endif

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
