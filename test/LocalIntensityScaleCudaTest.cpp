
#include <gtest/gtest.h>

#include "algorithm/LocalIntensityScaleCuda.h"
#include "algorithm/LocalIntensityScale.hpp"
#include "TestTools.hpp"


namespace {

#ifdef APR_USE_CUDA

    // ------------------------------------------------------------------------
    // TODO: REMOVE IT after dev.
    // ------------------------------------------------------------------------
    TEST(LocalIntensityScaleCudaTest, REMOVE_ME_AFTER_DEVELOPMENT) {
        int y_num = 2;
        int x_num = 3;
        int z_num = 2;
        PixelData<float> m(y_num, x_num, z_num, 0);
        PixelData<float> m2(y_num, x_num, z_num, 0);
        PixelData<float> m3(y_num, x_num, z_num,0);
        float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12};

        initFromZYXarray(m, dataIn);
        initFromZYXarray(m2, dataIn);
        initFromZYXarray(m3, dataIn);
        LocalIntensityScale lis;
        int off = 0;
        lis.calc_sat_mean_x(m, off);
        m.printMesh(1);
        calcMean(m3, off, MEAN_X_DIR);
        m3.printMesh(1);
//        lis.calc_sat_mean_y(m2, off);
//        m2.printMesh(1);


        compareMeshes(m3, m, 0.00000001);
    }
    // ------------------------------------------------------------------------

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_Y_DIR) {
        APRTimer timer(true);
        PixelData<float> m = getRandInitializedMesh<float>(22, 33, 22, 100, 3);

        LocalIntensityScale lis;
        for (int offset = 0; offset < 6; ++offset) {

            std::cout << " ============================== " << offset << std::endl;

            // Run on CPU
            PixelData<float> mCpu(m, true);
            timer.start_timer("CPU mean Y-DIR");
            lis.calc_sat_mean_y(mCpu, offset);
            timer.stop_timer();

            // Run on GPU
            PixelData<float> mGpu(m, true);
            timer.start_timer("GPU mean Y-DIR");
            calcMean(mGpu, offset, MEAN_Y_DIR);
            timer.stop_timer();

            // Compare results
            EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.01), 0);
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_X_DIR) {
        APRTimer timer(true);
        PixelData<float> m = getRandInitializedMesh<float>(22, 33, 22, 255);

        LocalIntensityScale lis;
        for (int offset = 0; offset < 6; ++offset) {

            std::cout << " ============================== " << offset << std::endl;

            // Run on CPU
            PixelData<float> mCpu(m, true);
            timer.start_timer("CPU mean X-DIR");
            lis.calc_sat_mean_x(mCpu, offset);
            timer.stop_timer();

            // Run on GPU
            PixelData<float> mGpu(m, true);
            timer.start_timer("GPU mean X-DIR");
            calcMean(mGpu, offset, MEAN_X_DIR);
            timer.stop_timer();

            // Compare results
            EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.001), 0);
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_Z_DIR) {
        APRTimer timer(true);
        using ImgType = float;
        PixelData<ImgType> m = getRandInitializedMesh<ImgType>(22, 33, 22, 255);

        LocalIntensityScale lis;
        for (int offset = 0; offset < 6; ++offset) {

            std::cout << " ============================== " << offset << std::endl;

            // Run on CPU
            PixelData<ImgType> mCpu(m, true);
            timer.start_timer("CPU mean Z-DIR");
            lis.calc_sat_mean_z(mCpu, offset);
            timer.stop_timer();

            // Run on GPU
            PixelData<ImgType> mGpu(m, true);
            timer.start_timer("GPU mean Z-DIR");
            calcMean(mGpu, offset, MEAN_Z_DIR);
            timer.stop_timer();

            // Compare results
            EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.000001), 0);
        }
    }


    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_WIHT_AND_WITHOUT_BOUNDARY_Y_DIR) {
        APRTimer timer(true);
        PixelData<float> m(4, 4, 1, 0);
        float dataIn[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        initFromZYXarray(m, dataIn);

        LocalIntensityScale lis;

        for (int boundary = 1; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            std::cout << "\n\n";
            for (int offset = 1; offset < 2; ++offset) {
                // Run on CPU
                PixelData<float> mCpuPadded;
                paddPixels(m, mCpuPadded, offset * boundary, offset * boundary, 0);
                timer.start_timer("CPU mean Y-DIR");
                lis.calc_sat_mean_y(mCpuPadded, offset);
                PixelData<float> mCpu;
                unpaddPixels(mCpuPadded, mCpu, m.y_num, m.x_num, m.z_num);
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean Y-DIR");
                calcMean(mGpu, offset, MEAN_Y_DIR, (boundary > 0));

                timer.stop_timer();

                // Compare results
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.01, 4), 0);
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_WIHT_AND_WITHOUT_BOUNDARY_X_DIR) {
        APRTimer timer(true);
        //PixelData<float> m(1, 13, 1, 0);
        //float dataIn[] = {1,2,3,4,5,6,7,8,9,10,11,12,13};
        //initFromZYXarray(m, dataIn);
        PixelData<float> m = getRandInitializedMesh<float>(31, 33, 13, 25, 10);

        LocalIntensityScale lis;

        for (int boundary = 0; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            for (int offset = 0; offset < 6; ++offset) {
                // Run on CPU
                PixelData<float> mCpuPadded;
                paddPixels(m, mCpuPadded, 0, offset * boundary, 0);
                timer.start_timer("CPU mean X-DIR");
                lis.calc_sat_mean_x(mCpuPadded, offset);
                PixelData<float> mCpu;
                unpaddPixels(mCpuPadded, mCpu, m.y_num, m.x_num, m.z_num);
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean X-DIR");
                calcMean(mGpu, offset, MEAN_X_DIR, (boundary > 0));
                timer.stop_timer();

                // Compare results
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.0000001), 0);
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_WIHT_AND_WITHOUT_BOUNDARY_Z_DIR) {
        APRTimer timer(true);
        PixelData<float> m(1, 1, 13, 0);
        float dataIn[] = {1,2,3,4,5,6,7,8,9,10,11,12,13};
        initFromZYXarray(m, dataIn);

        LocalIntensityScale lis;

        for (int boundary = 0; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            for (int offset = 0; offset < 6; ++offset) {
                // Run on CPU
                PixelData<float> mCpuPadded;
                paddPixels(m, mCpuPadded, 0, 0, offset * boundary);
                timer.start_timer("CPU mean Z-DIR");
                lis.calc_sat_mean_z(mCpuPadded, offset);
                PixelData<float> mCpu;
                unpaddPixels(mCpuPadded, mCpu, m.y_num, m.x_num, m.z_num);
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean Z-DIR");
                calcMean(mGpu, offset, MEAN_Z_DIR, (boundary > 0));
                timer.stop_timer();

                // Compare results
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.000001), 0);
            }
        }
    }


    // !!!!!!!!!!!!!!!!!!!!!!! NOT YET CHECKED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // TODO: See what these tests are doing and fix/change/remove them!

    TEST(LocalIntensityScaleCudaTest, 1D_Y_DIR) {
        {   // OFFSET=0

            PixelData<float> m(8, 1, 1, 0);
            float dataIn[] = {3,6,9,12,15,18,21,24};
            float expect[] = {3,6,9,12,15,18,21,24};

            initFromZYXarray(m, dataIn);

            calcMean(m, 0, MEAN_Y_DIR);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=1

            PixelData<float> m(8, 1, 1, 0);
            float dataIn[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float expect[] = {1.5, 2, 3, 4, 5, 6, 7, 7.5};

            initFromZYXarray(m, dataIn);

            calcMean(m, 1, MEAN_Y_DIR);

            ASSERT_TRUE(compare(m, expect, 0.05));
        }
        {   // OFFSET=2 (+symmetricity check)

            PixelData<float> m(8, 1, 1, 0);
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



    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_ALL_DIRS) {
        APRTimer timer(true);
        PixelData<float> m = getRandInitializedMesh<float>(33, 31, 13);

        LocalIntensityScale lis;
        for (int offset = 0; offset < 6; ++offset) {
            // Run on CPU
            PixelData<float> mCpu(m, true);
            timer.start_timer("CPU mean ALL-DIR");
            lis.calc_sat_mean_y(mCpu, offset);
            lis.calc_sat_mean_x(mCpu, offset);
            lis.calc_sat_mean_z(mCpu, offset);
            timer.stop_timer();

            // Run on GPU
            PixelData<float> mGpu(m, true);
            timer.start_timer("GPU mean ALL-DIR");
            calcMean(mGpu, offset);
            timer.stop_timer();

            // Compare results
            EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.01), 0);
        }
    }

    //@KG: The CPU code doesn't work for uint16 --> overflow will likely result.

//    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_ALL_DIRS_UINT16) {
//        APRTimer timer(true);
//        PixelData<uint16_t> m = getRandInitializedMesh<uint16_t>(33, 31, 13);
//
//        LocalIntensityScale lis;
//        for (int offset = 0; offset < 6; ++offset) {
//            // Run on CPU
//            PixelData<uint16_t> mCpu(m, true);
//            timer.start_timer("CPU mean ALL-DIR");
//            lis.calc_sat_mean_y(mCpu, offset);
//            lis.calc_sat_mean_x(mCpu, offset);
//            lis.calc_sat_mean_z(mCpu, offset);
//            timer.stop_timer();
//
//            // Run on GPU
//            PixelData<uint16_t> mGpu(m, true);
//            timer.start_timer("GPU mean ALL-DIR");
//            calcMean(mGpu, offset);
//            timer.stop_timer();
//
//            // Compare results
//            EXPECT_EQ(compareMeshes(mCpu, mGpu, 1), 0);
//        }
//    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_FULL_PIPELINE) {
        APRTimer timer(true);
        PixelData<float> m = getRandInitializedMesh<float>(31, 33, 13, 25, 10);

        APRParameters params;
        params.sigma_th = 1;
        params.sigma_th_max = 2;
        params.reflect_bc_lis = false; //#TODO: @KG: The CPU pipeline uses this to true, so needs to now be implimented.

        // Run on CPU
        PixelData<float> mCpu(m, true);
        PixelData<float> mCpuTemp(m, false);
        timer.start_timer("CPU LIS FULL");

        LocalIntensityScale localIntensityScale;

        localIntensityScale.get_local_intensity_scale(mCpu, mCpuTemp, params);
        timer.stop_timer();

        // Run on GPU
        PixelData<float> mGpu(m, true);
        PixelData<float> mGpuTemp(m, false);
        timer.start_timer("GPU LIS ALL-DIR");
        getLocalIntensityScale(mGpu, mGpuTemp, params);
        timer.stop_timer();

        // Compare results
        //EXPECT_EQ(compareMeshes(mCpuTemp, mGpuTemp, 0.01), 0); //this is not needed these values are not required.
        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.00001), 0);
    }


#endif // APR_USE_CUDA
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
