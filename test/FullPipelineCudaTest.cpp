
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


namespace {
#ifdef APR_USE_CUDA

    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS_LEVELS) {
        APRTimer timer(true);

        // Generate random mesh - keep it large enough to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim{333, 1000, 333};
        PixelData<ImageType> input_image = getRandInitializedMesh<ImageType>(dim, 13);
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
        PixelData<ImageType> grad_temp_GPU (grad_temp, true);
        PixelData<float> local_scale_temp_GPU(local_scale_temp, true);
        PixelData<float> local_scale_temp2_GPU(local_scale_temp2, true);

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
        EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU, 0), 0);
    }

    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS_LEVELS_PS) {
        APRTimer timer(true);

        // Generate random mesh - keep it large enough to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim{333, 1000, 333};
        PixelData<ImageType> input_image = getRandInitializedMesh<ImageType>(dim, 13);
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
        PixelData<ImageType> grad_temp_GPU (grad_temp, true);
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
        int levelMax = aprInfo.l_max - 1;
        int levelMin = aprInfo.l_min;
        std::vector<PixelData<uint8_t>> pct = PullingScheme::generateParticleCellTree(aprInfo);
        computeOvpcCuda(local_scale_temp_GPU, pct, levelMin, levelMax);
        timer.stop_timer();

        // Compare GPU vs CPU - expect exactly same result
        ASSERT_EQ(compareParticleCellTrees(ps.getParticleCellTree(), pct), 0);
    }

    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS_LEVELS_PS_LINEARACCESS) {
        APRTimer timer(true);

        // Generate random mesh - keep it large enough to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim{333, 1000, 333};
        PixelData<ImageType> input_image = getRandInitializedMesh<ImageType>(dim, 13);
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
        PixelData<ImageType> grad_temp_GPU (grad_temp, true);
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

        GenInfo aprInfo;
        aprInfo.init(input_image.getDimension());
        GenInfo giGpu;
        giGpu.init(input_image.getDimension());

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
        int levelMax = giGpu.l_max - 1;
        int levelMin = giGpu.l_min;
        std::vector<PixelData<uint8_t>> pct = PullingScheme::generateParticleCellTree(giGpu);
        computeOvpcCuda(local_scale_temp_GPU, pct, levelMin, levelMax);
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

    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS_LEVELS_GpuProcessingTask) {
        APRTimer timer(true);

        // Generate random mesh - keep it large enough to catch all possible computation errors
        using ImageType = float;
        constexpr PixelDataDim dim{333, 1000, 333};
        PixelData<ImageType> input_image = getRandInitializedMesh<ImageType>(dim, 99, 0, false);
        int maxLevel = ceil(std::log2(dim.maxDimSize()));

        PixelData<ImageType> grad_temp; // should be a down-sampled image
        grad_temp.initDownsampled(dim, 0, false);
        PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp.initDownsampled(dim,false);
        PixelData<float> local_scale_temp2;
        local_scale_temp2.initDownsampled(dim, false);

        PixelData<ImageType> grad_temp_GPU; // should be a down-sampled image
        grad_temp_GPU.initDownsampled(dim, 0, false);
        PixelData<float> local_scale_temp_GPU; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp_GPU.initDownsampled(dim, false);
        PixelData<float> local_scale_temp2_GPU;
        local_scale_temp2_GPU.initDownsampled(dim, false);

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
        PixelData<ImageType> mCpuImage(input_image, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
        ComputeGradient().get_gradient(mCpuImage, grad_temp, local_scale_temp, par);
        LocalIntensityScale().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
        LocalParticleCellSet().computeLevels(grad_temp, local_scale_temp, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
        timer.stop_timer();


        // Calculate pipeline on GPU
        PixelData<ImageType> mGpuImage(input_image, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");

        {
            GpuProcessingTask<ImageType> gpt(mGpuImage, local_scale_temp_GPU, par, 0, maxLevel);
            gpt.doAll();
        }
        timer.stop_timer();

        // Compare GPU vs CPU - expect exactly same result
        EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU, 0), 0);
    }
#endif // APR_USE_CUDA
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
