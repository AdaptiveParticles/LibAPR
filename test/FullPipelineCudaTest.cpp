
#include <gtest/gtest.h>

#include "algorithm/LocalIntensityScaleCuda.h"
#include "algorithm/LocalIntensityScale.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include "algorithm/PullingSchemeCuda.hpp"
#include "data_structures/APR/access/LinearAccessCuda.hpp"
#include "TestTools.hpp"
#include "data_structures/Mesh/PixelDataCuda.h"
#include "algorithm/APRConverter.hpp"
#include "misc/CudaTools.cuh"


namespace {
#ifdef APR_USE_CUDA

    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS) {
        APRTimer timer(true);

        // Generate random mesh of two sizes very small and reasonable large to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim1{4, 4, 3};
        constexpr PixelDataDim dim2{163, 123, 555};
        for (int d = 0; d <= 3; d++) {
            auto &dim = (d % 2 == 0) ? dim1 : dim2;
            PixelData<ImageType> input_image = (d/2 == 0) ? getRandInitializedMesh<ImageType>(dim, 13) :
                                                            getMeshWithBlobInMiddle<ImageType>(dim);

            // Initialize CPU data structures
            PixelData<ImageType> mCpuImage(input_image, true);
            PixelData<ImageType> grad_temp;
            grad_temp.initDownsampled(dim, 0, false);
            PixelData<float> local_scale_temp;
            local_scale_temp.initDownsampled(dim, false);
            PixelData<float> local_scale_temp2;
            local_scale_temp2.initDownsampled(dim, false);

            // Initialize GPU data structures to same values as CPU
            PixelData<ImageType> mGpuImage(input_image, true, true);
            PixelData<ImageType> grad_temp_GPU(grad_temp, true, true);
            PixelData<float> local_scale_temp_GPU(local_scale_temp, true, true);
            PixelData<float> local_scale_temp2_GPU(local_scale_temp2, true, true);

            // Prepare parameters
            APRParameters par;
            par.lambda = 3;
            par.Ip_th = 10;
            par.sigma_th = 0;
            par.sigma_th_max = 0;
            par.dx = 1;
            par.dy = 1;
            par.dz = 1;

            // Calculate pipeline on CPU
            timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
            ComputeGradient().get_gradient(mCpuImage, grad_temp, local_scale_temp, par);
            LocalIntensityScale().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
            timer.stop_timer();

            // Calculate pipeline on GPU
            timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");
            getGradient(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
            getLocalIntensityScale(local_scale_temp_GPU, local_scale_temp2_GPU, par);
            timer.stop_timer();

            // Compare GPU vs CPU - expect exactly same result
            EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU, 0), 0);
            EXPECT_EQ(compareMeshes(grad_temp, grad_temp_GPU, 0), 0);
        }
    }

    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS_LEVELS) {
        APRTimer timer(true);

        // Generate random mesh of two sizes very small and reasonable large to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim1{4, 4, 3};
        constexpr PixelDataDim dim2{163, 123, 555};
        for (int d = 0; d <= 3; d++) {
            auto &dim = (d%2 == 0) ? dim1 : dim2;
            PixelData<ImageType> input_image = (d/2 == 0) ? getRandInitializedMesh<ImageType>(dim, 13) :
                                               getMeshWithBlobInMiddle<ImageType>(dim);
            int maxLevel = ceil(std::log2(input_image.getDimension().maxDimSize()));

            // Initialize CPU data structures
            PixelData<ImageType> mCpuImage(input_image, true);
            PixelData<ImageType> grad_temp;
            grad_temp.initDownsampled(dim, 0, false);
            PixelData<float> local_scale_temp;
            local_scale_temp.initDownsampled(dim, false);
            PixelData<float> local_scale_temp2;
            local_scale_temp2.initDownsampled(dim, false);

            // Initialize GPU data structures to same values as CPU
            PixelData<ImageType> mGpuImage(input_image, true, false);
            PixelData<ImageType> grad_temp_GPU(grad_temp, true, false);
            PixelData<float> local_scale_temp_GPU(local_scale_temp, true, false);
            PixelData<float> local_scale_temp2_GPU(local_scale_temp2, true, false);

            // Prepare parameters
            APRParameters par;
            par.lambda = 3;
            par.Ip_th = 10;
            par.sigma_th = 0;
            par.sigma_th_max = 0;
            par.dx = 1;
            par.dy = 1;
            par.dz = 1;

            // Calculate pipeline on CPU
            timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
            ComputeGradient().get_gradient(mCpuImage, grad_temp, local_scale_temp, par);
            LocalIntensityScale().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
            LocalParticleCellSet().computeLevels(grad_temp, local_scale_temp, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
            timer.stop_timer();

            // Calculate pipeline on GPU
            timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");
            getGradient(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
            getLocalIntensityScale(local_scale_temp_GPU, local_scale_temp2_GPU, par);
            computeLevelsCuda(grad_temp_GPU, local_scale_temp_GPU, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
            timer.stop_timer();

            // Compare GPU vs CPU - expect exactly same result
            EXPECT_EQ(compareMeshes(grad_temp, grad_temp_GPU, 0), 0);
            EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU, 0), 0);
        }
    }

    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS_LEVELS_PS) {
        APRTimer timer(true);

        // Generate random mesh of two sizes very small and reasonable large to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim1{4, 4, 3};
        constexpr PixelDataDim dim2{163, 123, 555};
        for (int d = 0; d <= 3; d++) {
            auto &dim = (d % 2 == 0) ? dim1 : dim2;
            PixelData<ImageType> input_image = (d / 2 == 0) ? getRandInitializedMesh<ImageType>(dim, 13) :
                                                              getMeshWithBlobInMiddle<ImageType>(dim);
            int maxLevel = ceil(std::log2(input_image.getDimension().maxDimSize()));

            // Initialize CPU data structures
            PixelData<ImageType> mCpuImage(input_image, true);
            PixelData<ImageType> grad_temp;
            grad_temp.initDownsampled(dim, 0, false);
            PixelData<float> local_scale_temp;
            local_scale_temp.initDownsampled(dim, false);
            PixelData<float> local_scale_temp2;
            local_scale_temp2.initDownsampled(dim, false);

            // Initialize GPU data structures to same values as CPU
            PixelData<ImageType> mGpuImage(input_image, true);
            PixelData<ImageType> grad_temp_GPU(grad_temp, true);
            PixelData<float> local_scale_temp_GPU(local_scale_temp, true);
            PixelData<float> local_scale_temp2_GPU(local_scale_temp2, true);

            // Prepare parameters and APR info structures
            APRParameters par;
            par.lambda = 3;
            par.Ip_th = 10;
            par.sigma_th = 0;
            par.sigma_th_max = 0;
            par.dx = 1;
            par.dy = 1;
            par.dz = 1;

            GenInfo aprInfo;
            aprInfo.init(input_image.getDimension());

            // Calculate pipeline on CPU
            timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
            ComputeGradient().get_gradient(mCpuImage, grad_temp, local_scale_temp, par);
            LocalIntensityScale().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
            LocalParticleCellSet lpcs = LocalParticleCellSet();
            lpcs.computeLevels(grad_temp, local_scale_temp, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
            PullingScheme ps;
            ps.initialize_particle_cell_tree(aprInfo);
            lpcs.get_local_particle_cell_set(ps, local_scale_temp, local_scale_temp2, par);
            ps.pulling_scheme_main();
            timer.stop_timer();

            // Calculate pipeline on GPU
            timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");
            getGradient(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
            getLocalIntensityScale(local_scale_temp_GPU, local_scale_temp2_GPU, par);
            computeLevelsCuda(grad_temp_GPU, local_scale_temp_GPU, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
            auto pct = computeOvpcCuda(local_scale_temp_GPU, aprInfo);
            timer.stop_timer();

            // Compare GPU vs CPU - expect exactly same result
            ASSERT_EQ(compareParticleCellTrees(ps.getParticleCellTree(), pct), 0);
        }
    }




    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS_LEVELS_PS_LINEARACCESS) {
        APRTimer timer(true);

        // Generate random mesh of two sizes very small and reasonable large to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim1{4, 4, 3};
        constexpr PixelDataDim dim2{163, 123, 555};
        for (int d = 0; d <= 3; d++) {
            auto &dim = (d % 2 == 0) ? dim1 : dim2;
            PixelData<ImageType> input_image = (d / 2 == 0) ? getRandInitializedMesh<ImageType>(dim, 13) :
                                                              getMeshWithBlobInMiddle<ImageType>(dim);

            int maxLevel = ceil(std::log2(input_image.getDimension().maxDimSize()));

            // Initialize CPU data structures
            PixelData<ImageType> mCpuImage(input_image, true);
            PixelData<ImageType> grad_temp;
            grad_temp.initDownsampled(dim, 0, false);
            PixelData<float> local_scale_temp;
            local_scale_temp.initDownsampled(dim, false);
            PixelData<float> local_scale_temp2;
            local_scale_temp2.initDownsampled(dim, false);

            // Initialize GPU data structures to same values as CPU
            PixelData<ImageType> mGpuImage(input_image, true);
            PixelData<ImageType> grad_temp_GPU(grad_temp, true);
            PixelData<float> local_scale_temp_GPU(local_scale_temp, true);
            PixelData<float> local_scale_temp2_GPU(local_scale_temp2, true);

            // Prepare parameters and APR info structures
            APRParameters par;
            par.lambda = 3;
            par.Ip_th = 10;
            par.sigma_th = 0;
            par.sigma_th_max = 0;
            par.dx = 1;
            par.dy = 1;
            par.dz = 1;
            par.neighborhood_optimization = true;

            GenInfo aprInfo(input_image.getDimension());
            GenInfo giGpu(input_image.getDimension());

            // Calculate pipeline on CPU
            timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
            ComputeGradient().get_gradient(mCpuImage, grad_temp, local_scale_temp, par);
            LocalIntensityScale().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
            LocalParticleCellSet lpcs = LocalParticleCellSet();
            lpcs.computeLevels(grad_temp, local_scale_temp, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
            PullingScheme ps;
            ps.initialize_particle_cell_tree(aprInfo);
            lpcs.get_local_particle_cell_set(ps, local_scale_temp, local_scale_temp2, par);
            ps.pulling_scheme_main();
            LinearAccess linearAccess;
            linearAccess.genInfo = &aprInfo;

            linearAccess.initialize_linear_structure(par, ps.getParticleCellTree());
            timer.stop_timer();

            // Calculate pipeline on GPU
            timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");
            getGradient(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
            getLocalIntensityScale(local_scale_temp_GPU, local_scale_temp2_GPU, par);
            computeLevelsCuda(grad_temp_GPU, local_scale_temp_GPU, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
            auto pct = computeOvpcCuda(local_scale_temp_GPU, giGpu);
            auto linearAccessGpu = initializeLinearStructureCuda(giGpu, par, pct);
            timer.stop_timer();

            // Compare GPU vs CPU - expect exactly same result
            // Test if returned structures have same data
            EXPECT_EQ(compareParticles(linearAccessGpu.y_vec, linearAccess.y_vec), 0);
            EXPECT_EQ(compareParticles(linearAccessGpu.level_xz_vec, linearAccess.level_xz_vec), 0);
            EXPECT_EQ(compareParticles(linearAccessGpu.y_vec, linearAccess.y_vec), 0);

            EXPECT_EQ(aprInfo.total_number_particles, giGpu.total_number_particles);
            EXPECT_EQ(linearAccessGpu.y_vec.size(), linearAccess.y_vec.size());
        }
    }

    TEST(ComputeThreshold, FULL_PIPELINE_TEST_CPU_vs_GpuProcessingTask) {
        APRTimer timer(true);

        // Generate random mesh of two sizes very small and reasonable large to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim1{4, 4, 3};
        constexpr PixelDataDim dim2{1024,512,512};
        for (int d = 0; d <= 3; d++) {
            auto &dim = (d % 2 == 0) ? dim1 : dim2;
            PixelData<ImageType> input_image = (d / 2 == 0) ? getRandInitializedMesh<ImageType>(dim, 13) :
                                               getMeshWithBlobInMiddle<ImageType>(dim);

            int maxLevel = ceil(std::log2(dim.maxDimSize()));

            // Initialize CPU data structures
            PixelData<ImageType> mCpuImage(input_image, true);
            PixelData<ImageType> grad_temp;
            grad_temp.initDownsampled(dim, 0, false);
            PixelData<float> local_scale_temp;
            local_scale_temp.initDownsampled(dim, false);
            PixelData<float> local_scale_temp2;
            local_scale_temp2.initDownsampled(dim, false);

            // Initialize GPU data structures to same values as CPU
            PixelData<ImageType> mGpuImage(input_image, true);
            PixelData<float> local_scale_temp_GPU(local_scale_temp, false);

            // Prepare parameters
            APRParameters par;
            par.lambda = 3;
            par.Ip_th = 10;
            par.sigma_th = 0;
            par.sigma_th_max = 0;
            par.dx = 1;
            par.dy = 1;
            par.dz = 1;
            par.neighborhood_optimization = true;

            float bspline_offset = 0;

            GenInfo aprInfo(input_image.getDimension());
            GenInfo giGpu(input_image.getDimension());

            // Calculate pipeline on CPU
            timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
            ComputeGradient().get_gradient(mCpuImage, grad_temp, local_scale_temp, par);
            LocalIntensityScale().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
            LocalParticleCellSet lpcs = LocalParticleCellSet();
            ComputeGradient().applyParameters(grad_temp, local_scale_temp, local_scale_temp2, par, bspline_offset);
            lpcs.computeLevels(grad_temp, local_scale_temp, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
            PullingScheme ps;
            ps.initialize_particle_cell_tree(aprInfo);
            lpcs.get_local_particle_cell_set(ps, local_scale_temp, local_scale_temp2, par);
            ps.pulling_scheme_main();
            LinearAccess linearAccess;
            linearAccess.genInfo = &aprInfo;
            linearAccess.initialize_linear_structure(par, ps.getParticleCellTree());
            timer.stop_timer();


            // Calculate pipeline on GPU
            timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");
            GpuProcessingTask<ImageType> gpt(mGpuImage, local_scale_temp_GPU, par, bspline_offset, maxLevel);
            gpt.processOnGpu();
            auto linearAccessGpu = gpt.getDataFromGpu();
            giGpu.total_number_particles = linearAccessGpu.y_vec.size();
            cudaDeviceSynchronize();
            timer.stop_timer();

            // Compare GPU vs CPU - expect exactly same result
            EXPECT_EQ(compareParticles(linearAccessGpu.y_vec, linearAccess.y_vec), 0);
            EXPECT_EQ(compareParticles(linearAccessGpu.level_xz_vec, linearAccess.level_xz_vec), 0);
            EXPECT_EQ(compareParticles(linearAccessGpu.xz_end_vec, linearAccess.xz_end_vec), 0);

            EXPECT_EQ(aprInfo.total_number_particles, giGpu.total_number_particles);
            EXPECT_EQ(linearAccessGpu.y_vec.size(), linearAccess.y_vec.size());

        }
    }


    TEST(ComputeThreshold, FULL_PIPELINE_TEST_CPU_vs_GPU_via_APRConverter) {
        APRTimer timer(true);

        // Generate random mesh of two sizes very small and reasonable large to catch all possible computation errors
        using ImageType = uint16_t;
        std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere_original.tif";
        PixelData<ImageType> input_image_raw = TiffUtils::getMesh<uint16_t>(file_name);
        std::cout << input_image_raw << std::endl;

        // Prepare parameters
        APRParameters par;
        par.lambda = 2;
        par.Ip_th = -1;
        par.sigma_th = 234;
        par.sigma_th_max = 0;
        par.grad_th=10;
        par.dx = 1;
        par.dy = 1;
        par.dz = 1;
        par.neighborhood_optimization = true;
        par.auto_parameters = false;
        par.output_steps = false;
        par.neighborhood_optimization = true;
        par.sigma_th = 234;
        std::cout << par << std::endl;
        APR apr;
        APRConverter<uint16_t> converter;
        converter.par = par;
        converter.set_generate_linear(true);
        converter.set_sparse_pulling_scheme(false);
        converter.get_apr_cuda(apr, input_image_raw);
        std::cout << "APR CUDA total particles: " << apr.total_number_particles() << std::endl;

        APR apr2;
        APRConverter<uint16_t> converter2;
        converter2.par = par;
        converter2.set_generate_linear(true);
        converter2.set_sparse_pulling_scheme(false);
        converter2.get_apr_cpu(apr2, input_image_raw);
        std::cout << "APR  CPU total particles: " << apr2.total_number_particles() << std::endl;

        // Compare GPU vs CPU - expect exactly same result
        EXPECT_EQ(compareParticles(apr.linearAccess.y_vec, apr2.linearAccess.y_vec), 0);
        EXPECT_EQ(compareParticles(apr.linearAccess.level_xz_vec, apr2.linearAccess.level_xz_vec), 0);
        EXPECT_EQ(compareParticles(apr.linearAccess.xz_end_vec, apr2.linearAccess.xz_end_vec), 0);

        EXPECT_EQ(apr.total_number_particles(), apr2.total_number_particles());
        EXPECT_EQ(apr.linearAccess.y_vec.size(), apr2.linearAccess.y_vec.size());

    }
#endif // APR_USE_CUDA
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
