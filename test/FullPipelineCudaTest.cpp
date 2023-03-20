
#include <gtest/gtest.h>

#include "algorithm/LocalIntensityScaleCuda.h"
#include "algorithm/LocalIntensityScale.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include "TestTools.hpp"
#include "data_structures/Mesh/PixelDataCuda.h"
#include "algorithm/APRConverter.hpp"


namespace {
#ifdef APR_USE_CUDA

    TEST(ComputeThreshold, PIPELINE_TEST_GRADIENT_LIS_LEVELS) {
        APRTimer timer(true);

        // Generate random mesh - keep it large enough to catch all possible computation errors
        using ImageType = float;
        PixelData<ImageType> input_image = getRandInitializedMesh<ImageType>(1000, 1000, 1000, 13);
        int maxLevel = ceil(std::log2(input_image.getDimension().maxDimSize()));

        PixelData<ImageType> grad_temp; // should be a down-sampled image
        grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0, false);
        PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);
        PixelData<float> local_scale_temp2;
        local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);

        PixelData<ImageType> grad_temp_GPU; // should be a down-sampled image
        grad_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0, false);
        PixelData<float> local_scale_temp_GPU; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);
        PixelData<float> local_scale_temp2_GPU;
        local_scale_temp2_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);

        // Prepare parameters
        APRParameters par;
        par.lambda = 3;
        par.Ip_th = 10;
        par.sigma_th = 0;
        par.sigma_th_max = 0;
        par.dx = 1;
        par.dy = 1;
        par.dz = 1;

        // Calculate bspline on CPU
        PixelData<ImageType> mCpuImage(input_image, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
        ComputeGradient().get_gradient(mCpuImage, grad_temp, local_scale_temp, par);
        LocalIntensityScale().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
        LocalParticleCellSet().computeLevels(grad_temp, local_scale_temp, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImageType> mGpuImage(input_image, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");
        getGradient(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
        getLocalIntensityScale(local_scale_temp_GPU, local_scale_temp2_GPU, par);
        computeLevelsCuda(grad_temp_GPU, local_scale_temp_GPU, maxLevel, par.rel_error, par.dx, par.dy, par.dz);
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