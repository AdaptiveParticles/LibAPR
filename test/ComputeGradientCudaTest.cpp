
#include <gtest/gtest.h>

#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include "TestTools.hpp"

namespace {

#ifdef APR_USE_CUDA


    // ========================================================================
    // BSPLINE tests
    // ========================================================================

    template <typename T>
    class BsplineTest : public testing::Test {};
    TYPED_TEST_SUITE_P(BsplineTest);

    TYPED_TEST_P(BsplineTest, testBsplineInXdirCUDA) {
        APRTimer timer(false);

        std::vector<std::pair<int, int>> yzSizes = {{1,   1},
                                                    {32,  32},
                                                    {33,  33},
                                                    {44,  35},
                                                    {35,  44},
                                                    {255, 129}};

        for (auto &p: yzSizes) {
            int yLen = p.first;
            int zLen = p.second;
            // Run test with dimension in range much shorter than filter length to longer than filter length
            // (for lambda=3 and tolerance=0.00001 expected filter length k0=18)
            for (int xLen = 2; xLen < 22; ++xLen) {
                // Generate random mesh
                using ImgType = TypeParam;
                PixelData<ImgType> m = getRandInitializedMesh<ImgType>(yLen, xLen, zLen, 30, 10);

                // Filter parameters
                const float lambda = 3;
                const float tolerance = 0.0001;

                // Calculate bspline on CPU
                PixelData<ImgType> mCpu(m, true);
                timer.start_timer("CPU bspline");
                ComputeGradient().bspline_filt_rec_x(mCpu, lambda, tolerance);
                timer.stop_timer();

                // Calculate bspline on GPU
                PixelData<ImgType> mGpu(m, true);
                timer.start_timer("GPU bspline");
                cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_X_DIR);
                timer.stop_timer();

                // Compare GPU vs CPU
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
            }
        }
    }

    TYPED_TEST_P(BsplineTest, testBsplineInZdirCUDA) {
        APRTimer timer(false);

        std::vector<std::pair<int, int>> xySizes = {{1,   1},
                                                    {32,  32},
                                                    {33,  33},
                                                    {44,  35},
                                                    {35,  44},
                                                    {255, 129}};

        for (auto &p : xySizes) {
            int xLen = p.first;
            int yLen = p.second;
            // Run test with dimension in range much shorter than filter length to longer than filter length
            // (for lambda=3 and tolerance=0.00001 expected filter length k0=18)
            for (int zLen = 2; zLen < 22; ++zLen) {
                // Generate random mesh
                using ImgType = TypeParam;
                PixelData<ImgType> m = getRandInitializedMesh<ImgType>(yLen, xLen, zLen, 30, 10);

                // Filter parameters
                const float lambda = 3;
                const float tolerance = 0.0001;

                // Calculate bspline on CPU
                PixelData<ImgType> mCpu(m, true);
                timer.start_timer("CPU bspline");
                ComputeGradient().bspline_filt_rec_z(mCpu, lambda, tolerance);
                timer.stop_timer();

                // Calculate bspline on GPU
                PixelData<ImgType> mGpu(m, true);
                timer.start_timer("GPU bspline");
                cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_Z_DIR);
                timer.stop_timer();

                // Compare GPU vs CPU
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
            }
        }
    }

    TYPED_TEST_P(BsplineTest, testBsplineInYdirCUDA) {
        APRTimer timer(false);

        std::vector<std::pair<int, int>> xzSizes = {{1,   1},
                                                    {32,  32},
                                                    {33,  33},
                                                    {44,  35},
                                                    {35,  44},
                                                    {255, 129}};

        for (auto &p : xzSizes) {
            int xLen = p.first;
            int zLen = p.second;
            // Run test with dimension in range much shorter than filter length to longer than filter length
            // (for lambda=3 and tolerance=0.00001 expected filter length k0=18)
            for (int yLen = 2; yLen < 22; ++yLen) {
                // Generate random mesh
                using ImgType = TypeParam;
                PixelData<ImgType> m = getRandInitializedMesh<ImgType>(yLen, xLen, zLen, 30, 10);

                // Filter parameters
                const float lambda = 3;
                const float tolerance = 0.0001;

                // Calculate bspline on CPU
                PixelData<ImgType> mCpu(m, true);
                timer.start_timer("CPU bspline");
                ComputeGradient().bspline_filt_rec_y(mCpu, lambda, tolerance);
                timer.stop_timer();

                // Calculate bspline on GPU
                PixelData<ImgType> mGpu(m, true);
                timer.start_timer("GPU bspline");
                cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_Y_DIR);
                timer.stop_timer();

                //Compare GPU vs CPU
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
            }
        }
    }

    REGISTER_TYPED_TEST_SUITE_P(BsplineTest, testBsplineInXdirCUDA, testBsplineInZdirCUDA, testBsplineInYdirCUDA);
    using ImgTypes = ::testing::Types< float, uint16_t, int16_t, uint8_t>;
    INSTANTIATE_TYPED_TEST_SUITE_P(Testing, BsplineTest, ImgTypes);

    TEST(ComputeBspineTest, BSPLINE_FULL_XYZ_DIR_CUDA) {
        APRTimer timer(false);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(127, 128, 129, 100, 10);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001; // as defined in get_smooth_bspline_3D

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().get_smooth_bspline_3D(mCpu, lambda);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_ALL_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
    }


    // ========================================================================
    // INV. BSPLINE tests
    // ========================================================================

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_X_RND_CUDA) {
        APRTimer timer(false);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(127, 61, 66, 100, 10);

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_x(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_X_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Z_RND_CUDA) {
        APRTimer timer(false);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(128, 61, 66, 100, 10);

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_z(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_Z_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Y_RND_CUDA) {
        APRTimer timer(false);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(127, 61, 71, 100, 10);

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_y(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_Y_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_FULL_XYZ_DIR_RND_CUDA) {
        APRTimer timer(false);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(32,32,32,100, 10);

        // Calculate bspline on CPU
        PixelData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_y(mCpu);
        ComputeGradient().calc_inv_bspline_x(mCpu);
        ComputeGradient().calc_inv_bspline_z(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_ALL_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
    }

    // ========================================================================
    // Downsampled gradient
    // ========================================================================

    TEST(ComputeGradientTest, GPU_VS_CPU_DOWNSAMPLE_GRADIENT_ON_RANDOM_VALUES) {
        APRTimer timer(false);

        // Generate random mesh
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(31, 32, 33, 100);

        // Calculate gradient on CPU
        PixelData<ImgType> grad;
        grad.initDownsampled(m, 0);
        timer.start_timer("CPU gradient");
        ComputeGradient().calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
        timer.stop_timer();

        // Calculate gradient on GPU
        PixelData<ImgType> gradCuda;
        gradCuda.initDownsampled(m, 0);
        timer.start_timer("GPU gradient");
        cudaDownsampledGradient(m, gradCuda, 1, 1, 1);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(grad, gradCuda, 0), 0);
    }


    // ========================================================================
    // Full pipeline/gradient tests
    // ========================================================================

    TEST(ComputeThreshold, FULL_GRADIENT_TEST) {
        APRTimer timer(false);

        // Generate random mesh
        using ImageType = uint16_t;
        PixelData<ImageType> input_image = getRandInitializedMesh<ImageType>(33, 35, 37, 15, 20);
        PixelData<ImageType> &image_temp = input_image;

        PixelData<ImageType> grad_temp; // should be a down-sampled image
        grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0, false);
        PixelData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);
        PixelData<float> local_scale_temp2;
        local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);

        PixelData<ImageType> grad_temp_GPU; // should be a down-sampled image
        grad_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0, false);
        PixelData<float> local_scale_temp_GPU; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, true);
        PixelData<float> local_scale_temp2_GPU;
        local_scale_temp2_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, false);

        APRParameters par;
        par.lambda = 3;
        par.Ip_th = 10;
        par.dx = 1;
        par.dy = 1;
        par.dz = 1;

        // Calculate bspline on CPU
        PixelData<ImageType> mCpuImage(image_temp, true);

        ComputeGradient computeGradient;

        timer.start_timer(">>>>>>>>>>>>>>>>> CPU gradient");
        computeGradient.get_gradient(mCpuImage, grad_temp, local_scale_temp, par);
        timer.stop_timer();

        // Calculate bspline on GPU
        PixelData<ImageType> mGpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> GPU gradient");
        getGradient(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpuImage, mGpuImage, 0), 0);
        EXPECT_EQ(compareMeshes(grad_temp, grad_temp_GPU, 0), 0);
        EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU, 0), 0);
    }

#endif // APR_USE_CUDA

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
