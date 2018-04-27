//
// Created by Krzysztof Gonciarz on 4/10/18.
//

#include <gtest/gtest.h>
#include "data_structures/Mesh/MeshData.hpp"
#include "algorithm/LocalIntensityScale.hpp"
#include "algorithm/LocalIntensityScaleCuda.h"
#include "algorithm/APRConverter.hpp"
#include "TestTools.hpp"


namespace {
    TEST(LocalIntensityScaleTest, 1D_Y_DIR) {
        {   // OFFSET=0

            MeshData<float> m(8, 1, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {3,6,9,12,15,18,21,24};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_y(m, 0);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=1

            MeshData<float> m(8, 1, 1, 0);
            float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float expect[] = {1.5, 2, 3, 4, 5, 6, 7, 7.5};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_y(m, 1);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=2 (+symmetricity check)

            MeshData<float> m(8, 1, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {6, 7.5, 9, 12, 15, 18, 19.5, 21};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_y(m, 2);

            ASSERT_TRUE(compare(m, expect, 0.05));

            // check if data in opposite order gives same result
            float dataIn2[] = {24,21,18,15,12,9,6,3};
            float expect2[] = {21, 19.5, 18, 15,12, 9, 7.5, 6};

            initFromZYXarray(m, dataIn2);

            lis.calc_sat_mean_y(m, 2);

            ASSERT_TRUE(compare(m, expect2, 0.05));
        }
    }

    TEST(LocalIntensityScaleTest, 1D_X_DIR) {
        {   // OFFSET=0

            MeshData<float> m(1, 8, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {3,6,9,12,15,18,21,24};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_x(m, 0);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=1

            MeshData<float> m(1, 8, 1, 0);
            float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float expect[] = {1.5, 2, 3, 4, 5, 6, 7, 7.5};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_x(m, 1);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=2 (+symmetricity check)

            MeshData<float> m(1, 8, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {6, 7.5, 9, 12, 15, 18, 19.5, 21};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_x(m, 2);

            ASSERT_TRUE(compare(m, expect, 0.05));

            // check if data in opposite order gives same result
            float dataIn2[] = {24,21,18,15,12,9,6,3};
            float expect2[] = {21, 19.5, 18, 15,12, 9, 7.5, 6};

            initFromZYXarray(m, dataIn2);

            lis.calc_sat_mean_x(m, 2);

            ASSERT_TRUE(compare(m, expect2, 0.05));
        }
    }

    TEST(LocalIntensityScaleTest, 1D_Z_DIR) {
        {   // OFFSET=0

            MeshData<float> m(1, 1, 8, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {3,6,9,12,15,18,21,24};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_z(m, 0);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=1

            MeshData<float> m(1, 1, 8, 0);
            float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float expect[] = {1.5, 2, 3, 4, 5, 6, 7, 7.5};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_z(m, 1);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=2 (+symmetricity check)

            MeshData<float> m(1, 1, 8, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {6, 7.5, 9, 12, 15, 18, 19.5, 21};

            initFromZYXarray(m, dataIn);

            LocalIntensityScale lis;
            lis.calc_sat_mean_z(m, 2);

            ASSERT_TRUE(compare(m, expect, 0.05));

            // check if data in opposite order gives same result
            float dataIn2[] = {24,21,18,15,12,9,6,3};
            float expect2[] = {21, 19.5, 18, 15,12, 9, 7.5, 6};

            initFromZYXarray(m, dataIn2);

            lis.calc_sat_mean_z(m, 2);

            ASSERT_TRUE(compare(m, expect2, 0.05));
        }
    }


// ============================================================================
// ====================       CUDA IMPL TESTS     =============================
// ============================================================================

#ifdef APR_USE_CUDA

    TEST(LocalIntensityScaleCudaTest, 1D_Y_DIR) {
        {   // OFFSET=0

            MeshData<float> m(8, 1, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {3,6,9,12,15,18,21,24};

            initFromZYXarray(m, dataIn);

            calcMean(m, 0, MEAN_Y_DIR);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=1

            MeshData<float> m(8, 1, 1, 0);
            float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float expect[] = {1.5, 2, 3, 4, 5, 6, 7, 7.5};

            initFromZYXarray(m, dataIn);

            calcMean(m, 1, MEAN_Y_DIR);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=2 (+symmetricity check)

            MeshData<float> m(8, 1, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {6, 7.5, 9, 12, 15, 18, 19.5, 21};

            initFromZYXarray(m, dataIn);

            calcMean(m, 2, MEAN_Y_DIR);

            ASSERT_TRUE(compare(m, expect, 0.05));

            // check if data in opposite order gives same result
            float dataIn2[] = {24,21,18,15,12,9,6,3};
            float expect2[] = {21, 19.5, 18, 15,12, 9, 7.5, 6};

            initFromZYXarray(m, dataIn2);

            calcMean(m, 2, MEAN_Y_DIR);

            ASSERT_TRUE(compare(m, expect2, 0.05));
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_Y_DIR) {
        APRTimer timer(true);
        MeshData<float> m = getRandInitializedMesh<float>(33, 31, 13);

        LocalIntensityScale lis;
        for (int offset = 0; offset < 6; ++offset) {
            // Run on CPU
            MeshData<float> mCpu(m, true);
            timer.start_timer("CPU mean Y-DIR");
            lis.calc_sat_mean_y(mCpu, offset);
            timer.stop_timer();

            // Run on GPU
            MeshData<float> mGpu(m, true);
            timer.start_timer("GPU mean Y-DIR");
            calcMean(mGpu, offset, MEAN_Y_DIR);
            timer.stop_timer();

            // Compare results
            EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.01), 0);
        }
    }

    TEST(LocalIntensityScaleCudaTest, 1GPU_VS_CPU_X_DIR) {
        APRTimer timer(true);
        MeshData<float> m = getRandInitializedMesh<float>(33, 31, 13);

        LocalIntensityScale lis;
        for (int offset = 0; offset < 6; ++offset) {
            // Run on CPU
            MeshData<float> mCpu(m, true);
            timer.start_timer("CPU mean X-DIR");
            lis.calc_sat_mean_x(mCpu, offset);
            timer.stop_timer();

            // Run on GPU
            MeshData<float> mGpu(m, true);
            timer.start_timer("GPU mean X-DIR");
            calcMean(mGpu, offset, MEAN_X_DIR);
            timer.stop_timer();

            // Compare results
            EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.01), 0);
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_Z_DIR) {
        APRTimer timer(true);
        MeshData<float> m = getRandInitializedMesh<float>(33, 31, 13);

        LocalIntensityScale lis;
        for (int offset = 0; offset < 6; ++offset) {
            // Run on CPU
            MeshData<float> mCpu(m, true);
            timer.start_timer("CPU mean Z-DIR");
            lis.calc_sat_mean_z(mCpu, offset);
            timer.stop_timer();

            // Run on GPU
            MeshData<float> mGpu(m, true);
            timer.start_timer("GPU mean Z-DIR");
            calcMean(mGpu, offset, MEAN_Z_DIR);
            timer.stop_timer();

            // Compare results
            EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.01), 0);
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_ALL_DIRS) {
        APRTimer timer(true);
        MeshData<float> m = getRandInitializedMesh<float>(33, 31, 13);

        LocalIntensityScale lis;
        for (int offset = 0; offset < 6; ++offset) {
            // Run on CPU
            MeshData<float> mCpu(m, true);
            timer.start_timer("CPU mean ALL-DIR");
            lis.calc_sat_mean_y(mCpu, offset);
            lis.calc_sat_mean_x(mCpu, offset);
            lis.calc_sat_mean_z(mCpu, offset);
            timer.stop_timer();

            // Run on GPU
            MeshData<float> mGpu(m, true);
            timer.start_timer("GPU mean ALL-DIR");
            calcMean(mGpu, offset);
            timer.stop_timer();

            // Compare results
            EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.01), 0);
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_FULL_PIPELINE) {
        APRTimer timer(true);
        MeshData<float> m = getRandInitializedMesh<float>(119, 33, 31, 10);

        APRParameters params;
        params.sigma_th = 1;
        params.sigma_th_max = 2;

        // Run on CPU
        MeshData<float> mCpu(m, true);
        MeshData<float> mCpuTemp(m, false);
        timer.start_timer("CPU LIS FULL");
        APRConverter<float>().get_local_intensity_scale(mCpu, mCpuTemp, params);
        timer.stop_timer();

        // Run on GPU
        MeshData<float> mGpu(m, true);
        MeshData<float> mGpuTemp(m, false);
        timer.start_timer("GPU LIS ALL-DIR");
        getLocalIntensityScale(mGpu, mGpuTemp, params);
        timer.stop_timer();

        // Compare results
        EXPECT_EQ(compareMeshes(mCpuTemp, mGpuTemp, 0.01), 0);
        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.01), 0);
    }

#endif // APR_USE_CUDA

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
