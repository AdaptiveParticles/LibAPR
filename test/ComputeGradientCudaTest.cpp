
#include <gtest/gtest.h>

#include "data_structures/Mesh/PixelData.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include "TestTools.hpp"

namespace {

#ifdef APR_USE_CUDA

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
                EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
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
                EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
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
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.0001, 2), 0);
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
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }
#endif // APR_USE_CUDA

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
