
#include <gtest/gtest.h>

#include "algorithm/LocalIntensityScaleCuda.h"
#include "algorithm/LocalIntensityScale.hpp"
#include "TestTools.hpp"


namespace {

#ifdef APR_USE_CUDA

    TEST(LocalIntensityScaleCudaTest, CPU_AND_GPU_TEST_X_DIR_VS_MANUALLY_CALCULATED_VALUES) {
        // Belows data is precomputed for x-len = 5 (and maximum offset = 4) so do not change these numbers!
        constexpr PixelDataDim const dim{1, 5, 1};
        float expectedData[2][5][dim.x] = {
                        {   // with no boundary values
                            {1.00, 2.00, 3.00, 4.00, 5.00},  // offset = 0
                            {1.50, 2.00, 3.00, 4.00, 4.50},  // offset = 1
                            {2.00, 2.50, 3.00, 3.50, 4.00},  // offset = 2
                            {2.50, 3.00, 3.00, 3.00, 3.50},  // offset = 3
                            {3.00, 3.00, 3.00, 3.00, 3.00}   // offset = 4
                        },
                        {   // with boundary values
                            {1.00, 2.00, 3.00, 4.00, 5.00},
                            {1.66, 2.00, 3.00, 4.00, 4.33},
                            {2.20, 2.40, 3.00, 3.60, 3.80},
                            {2.71, 2.85, 3.00, 3.14, 3.28},
                            {3.22, 3.11, 3.00, 2.88, 2.77}
                        }
                    };

        APRTimer timer(false); // set to true to see timings

        PixelData<float> m(dim);
        float dataIn[] = {1, 2, 3, 4, 5};
        initFromZYXarray(m, dataIn);

        LocalIntensityScale lis;

        for (int boundary = 0; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            for (int offset = 0; offset <= 4; ++offset) {
//                std::cout << "OFFSET=" << offset << " boundary=" << (boundary > 0) << std::endl;

                // Run on CPU
                PixelData<float> mCpu(m, true);
                timer.start_timer("CPU mean X-DIR");
                lis.calc_sat_mean_x(mCpu, offset, (boundary > 0));
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean X-DIR");
                calcMean(mGpu, offset, MEAN_X_DIR, (boundary > 0));
                timer.stop_timer();

                // Compare results
                PixelData<float> expected(dim);
                initFromZYXarray(expected, expectedData[boundary][offset]);
                EXPECT_EQ(compareMeshes(expected, mGpu, 0.01), 0);
                EXPECT_EQ(compareMeshes(expected, mCpu, 0.01), 0);

                // Also GPU and CPU should give exactly same output
                EXPECT_EQ(compareMeshes(mGpu, mCpu, 0), 0);
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_WITH_AND_WITHOUT_BOUNDARY_X_DIR_RANDOM_VALUES) {
        APRTimer timer(false);

        constexpr PixelDataDim const dim{49, 53, 51};
        PixelData<float> m = getRandInitializedMesh<float>(dim, 50, 10);

        LocalIntensityScale lis;

        for (int boundary = 0; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            for (int offset = 0; offset <= 6; ++offset) {
//                std::cout << "------------- OFFSET=" << offset << " boundary=" << (boundary > 0) << std::endl;

                PixelData<float> mCpu;
                mCpu.init(m);
                mCpu.copyFromMesh(m);
                timer.start_timer("CPU mean X-DIR");
                lis.calc_sat_mean_x(mCpu, offset, (boundary > 0));
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean X-DIR");
                calcMean(mGpu, offset, MEAN_X_DIR, (boundary > 0));
                timer.stop_timer();

                // Compare results
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0); // Expect exactly same results
            }
        }
    }

    /**
     * Generate input and expected output using easy brute force approach.
     * When comparing vs CPU or GPU outputs there is small error expected since little difference in order of float
     * operations.
     * @tparam T - type of generated data
     * @param len - length
     * @param offset - offset for which expected output should be calculated
     * @param boundary - use boundary?
     * @param useRandomNumbers - use random numbers or if false then index numbers in buffers [1..len]
     * @return tuple of [input, expectedOutput]
     */
    template <typename T>
    auto generateInputAndExpected(int len, int offset, bool boundary, bool useRandomNumbers) {
        std::vector<T> input(len);
        std::vector<T> expected(len);

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(0.0, 10.0);

        // Feel input and calculate expected data
        for (int i = 0; i < len; ++i) input[i] = useRandomNumbers ? dist(mt) : i + 1;

        for (int i = 0; i < len; ++i) {
            int count = 0;
            T sum = 0;
            for (int x = i - offset; x <= i + offset; ++x) {
                int currIdx = x;
                if (boundary) {
                    currIdx = abs(x);
                    if (currIdx > len - 1) currIdx = (len - 1) - (currIdx - (len - 1));
                }

                if (currIdx < 0 || currIdx >= len) continue;

                sum += input[currIdx];
                count++;
            }
            expected[i] = sum / count;
        }
        return std::make_tuple(input, expected);
    }

    TEST(LocalIntensityScaleCudaTest, GPU_CPU_VS_PRECOMPUTED_VALUES_X_DIR) {
        // Input params
        using T = float;

        for (int b = 0; b <= 1; b++) {
            for (int len = 5; len <= 45; len += 20) {
                for (int offset = 0; offset <= 6 && offset < len; offset++) {
                    for (int r = 0; r <= 1; r++) {
                        bool hasBoundary = b > 0;
                        bool useRandomNumbers = r > 0;
//                        std::cout << "========================> len=" << len << " offset=" << offset << " hasBoundary=" << hasBoundary << " useRandomNumbers=" << useRandomNumbers << std::endl;

                        auto t = generateInputAndExpected<T>(len, offset, hasBoundary, useRandomNumbers);
                        auto input = std::get<0>(t);
                        auto expected = std::get<1>(t);
                        PixelData<T> m(1, len, 1, 0);
                        initFromZYXarray(m, input.data());
                        PixelData<T> expectedMesh(1, len, 1, 0);
                        initFromZYXarray(expectedMesh, expected.data());

                        APRTimer timer(false);
                        LocalIntensityScale lis;

                        // Run on CPU old-impl
                        timer.start_timer("CPU X-DIR");
                        PixelData<T> mCpu(m, true);
                        lis.calc_sat_mean_x(mCpu, offset, hasBoundary);
                        timer.stop_timer();

                        // Run on GPU
                        PixelData<T> mGpu(m, true);
                        timer.start_timer("GPU X-DIR");
                        calcMean(mGpu, offset, MEAN_X_DIR, hasBoundary);
                        timer.stop_timer();

                        // expectedMesh because of different order of calculation will have small floating-point differences
                        // comparing to CPU or GPU fast implementation, anyway GPU and CPU should have exactly same values!
                        EXPECT_EQ(compareMeshes(expectedMesh, mGpu, 0.00001), 0) << "---!!!!!!--- GPU values does not match";
                        EXPECT_EQ(compareMeshes(expectedMesh, mCpu, 0.00001), 0) << "---!!!!!!--- CPU values does not match";
                        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0) << "---!!!!!!--- CPU vs GPU values does not match";
                    }
                }
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, CPU_AND_GPU_TEST_Z_DIR_VS_MANUALLY_CALCULATED_VALUES) {
        // Belows data is precomputed for x-len = 5 (and maximum offset = 4) so do not change these numbers!
        constexpr PixelDataDim const dim{1, 1, 5};
        float expectedData[2][5][dim.z] = {
                {   // with no boundary values
                        {1.00, 2.00, 3.00, 4.00, 5.00},  // offset = 0
                        {1.50, 2.00, 3.00, 4.00, 4.50},  // offset = 1
                        {2.00, 2.50, 3.00, 3.50, 4.00},  // offset = 2
                        {2.50, 3.00, 3.00, 3.00, 3.50},  // offset = 3
                        {3.00, 3.00, 3.00, 3.00, 3.00}   // offset = 4
                },
                {   // with boundary values
                        {1.00, 2.00, 3.00, 4.00, 5.00},
                        {1.66, 2.00, 3.00, 4.00, 4.33},
                        {2.20, 2.40, 3.00, 3.60, 3.80},
                        {2.71, 2.85, 3.00, 3.14, 3.28},
                        {3.22, 3.11, 3.00, 2.88, 2.77}
                }
        };

        APRTimer timer(false); // set to true to see timings

        PixelData<float> m(dim);
        float dataIn[] = {1, 2, 3, 4, 5};
        initFromZYXarray(m, dataIn);

        LocalIntensityScale lis;

        for (int boundary = 0; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            for (int offset = 0; offset <= 4; ++offset) {
//                std::cout << "------------------ OFFSET=" << offset << " boundary=" << (boundary > 0) << std::endl;

                // Run on CPU
                PixelData<float> mCpu(m, true);
                timer.start_timer("CPU mean X-DIR");
                lis.calc_sat_mean_z(mCpu, offset, (boundary > 0));
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean X-DIR");
                calcMean(mGpu, offset, MEAN_Z_DIR, (boundary > 0));
                timer.stop_timer();

                // Compare results
                PixelData<float> expected(dim);
                initFromZYXarray(expected, expectedData[boundary][offset]);
                EXPECT_EQ(compareMeshes(expected, mGpu, 0.01), 0);
                EXPECT_EQ(compareMeshes(expected, mCpu, 0.01), 0);

                // Also GPU and CPU should give exactly same output
                EXPECT_EQ(compareMeshes(mGpu, mCpu, 0), 0);
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_WITH_AND_WITHOUT_BOUNDARY_Z_DIR_RANDOM_VALUES) {
        APRTimer timer(false);

        constexpr PixelDataDim const dim{49, 51, 53};
        PixelData<float> m = getRandInitializedMesh<float>(dim, 50, 10);

        LocalIntensityScale lis;

        for (int boundary = 0; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            for (int offset = 0; offset <= 6; ++offset) {
//                std::cout << "---------------- OFFSET=" << offset << " boundary=" << (boundary > 0) << std::endl;

                PixelData<float> mCpu;
                mCpu.init(m);
                mCpu.copyFromMesh(m);
                timer.start_timer("CPU mean Z-DIR");
                lis.calc_sat_mean_z(mCpu, offset, (boundary > 0));
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean Z-DIR");
                calcMean(mGpu, offset, MEAN_Z_DIR, (boundary > 0));
                timer.stop_timer();

                // Compare results
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_CPU_VS_PRECOMPUTED_VALUES_Z_DIR) {
        // Input params
        using T = float;

        for (int b = 0; b <= 1; b++) {
            for (int len = 5; len <= 45; len += 20) {
                for (int offset = 0; offset <= 6 && offset < len; offset++) {
                    for (int r = 0; r <= 1; r++) {
                        bool hasBoundary = b > 0;
                        bool useRandomNumbers = r > 0;
//                        std::cout << "========================> len=" << len << " offset=" << offset << " hasBoundary=" << hasBoundary << " useRandomNumbers=" << useRandomNumbers << std::endl;

                        auto t = generateInputAndExpected<T>(len, offset, hasBoundary, useRandomNumbers);
                        auto input = std::get<0>(t);
                        auto expected = std::get<1>(t);
                        PixelData<T> m(1, 1, len, 0);
                        initFromZYXarray(m, input.data());
                        PixelData<T> expectedMesh(1, 1, len, 0);
                        initFromZYXarray(expectedMesh, expected.data());

                        APRTimer timer(false);
                        LocalIntensityScale lis;

                        // Run on CPU old-impl
                        timer.start_timer("CPU Z-DIR");
                        PixelData<T> mCpu(m, true);
                        lis.calc_sat_mean_z(mCpu, offset, hasBoundary);
                        timer.stop_timer();

                        // Run on GPU
                        PixelData<T> mGpu(m, true);
                        timer.start_timer("GPU Z-DIR");
                        calcMean(mGpu, offset, MEAN_Z_DIR, hasBoundary);
                        timer.stop_timer();

                        // expectedMesh because of different order of calculation will have small floating-point differences
                        // comparing to CPU or GPU fast implementation, anyway GPU and CPU should have exactly same values!
                        EXPECT_EQ(compareMeshes(expectedMesh, mGpu, 0.00001), 0) << "---!!!!!!--- GPU values does not match";
                        EXPECT_EQ(compareMeshes(expectedMesh, mCpu, 0.00001), 0) << "---!!!!!!--- CPU values does not match";
                        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.0), 0) << "---!!!!!!--- CPU vs GPU values does not match";
                    }
                }
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, CPU_AND_GPU_TEST_Y_DIR_VS_MANUALLY_CALCULATED_VALUES) {
        // Belows data is precomputed for y_len = 5 (and maximum offset = 4) so do not change these numbers!
        constexpr PixelDataDim const dim{5, 1, 1};
        float expectedData[2][5][dim.y] = {
                {   // with no boundary values
                        {1.00, 2.00, 3.00, 4.00, 5.00},  // offset = 0
                        {1.50, 2.00, 3.00, 4.00, 4.50},  // offset = 1
                        {2.00, 2.50, 3.00, 3.50, 4.00},  // offset = 2
                        {2.50, 3.00, 3.00, 3.00, 3.50},  // offset = 3
                        {3.00, 3.00, 3.00, 3.00, 3.00}   // offset = 4
                },
                {   // with boundary values
                        {1.00, 2.00, 3.00, 4.00, 5.00},
                        {1.66, 2.00, 3.00, 4.00, 4.33},
                        {2.20, 2.40, 3.00, 3.60, 3.80},
                        {2.71, 2.85, 3.00, 3.14, 3.28},
                        {3.22, 3.11, 3.00, 2.88, 2.77}
                }
        };

        APRTimer timer(false); // set to true to see timings

        PixelData<float> m(dim);
        float dataIn[] = {1, 2, 3, 4, 5};
        initFromZYXarray(m, dataIn);

        LocalIntensityScale lis;

        for (int boundary = 0; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            for (int offset = 0; offset <= 4; ++offset) {
                // std::cout << "------------- OFFSET=" << offset << " boundary=" << (boundary > 0) << std::endl;

                // Run on CPU
                PixelData<float> mCpu(m, true);
                timer.start_timer("CPU mean Y-DIR");
                lis.calc_sat_mean_y(mCpu, offset, (boundary > 0));
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean Y-DIR");
                calcMean(mGpu, offset, MEAN_Y_DIR, (boundary > 0));
                timer.stop_timer();

                // Compare results
                PixelData<float> expected(dim);
                initFromZYXarray(expected, expectedData[boundary][offset]);
                EXPECT_EQ(compareMeshes(expected, mGpu, 0.01), 0);
                EXPECT_EQ(compareMeshes(expected, mCpu, 0.01), 0);

                // Also GPU and CPU should give exactly same output
                EXPECT_EQ(compareMeshes(mGpu, mCpu, 0), 0);
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_WITH_AND_WITHOUT_BOUNDARY_Y_DIR_RANDOM_VALUES) {
        APRTimer timer(false);

        constexpr PixelDataDim const dim{49, 51, 53};
        PixelData<float> m = getRandInitializedMesh<float>(dim, 2, 0,false);

        LocalIntensityScale lis;

        for (int boundary = 0; boundary <= 1; ++ boundary) {
            // boundary = 0 there is no reflected boundary
            // boudnary = 1 there is boundary reflect
            for (int offset = 0; offset <= 6; ++offset) {
//                std::cout << "---------------- OFFSET=" << offset << " boundary=" << (boundary > 0) << std::endl;

                PixelData<float> mCpu(m, true);
                timer.start_timer("CPU mean Y-DIR");
                lis.calc_sat_mean_y(mCpu, offset, (boundary > 0));
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean Y-DIR");
                calcMean(mGpu, offset, MEAN_Y_DIR, (boundary > 0));
                timer.stop_timer();

                // Compare results
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_CPU_VS_PRECOMPUTED_VALUES_Y_DIR) {
        // Input params
        using T = float;

        for (int b = 0; b <= 1; b++) {
            for (int len = 5; len <= 45; len += 20) {
                for (int offset = 0; offset <= 6 && offset < len; offset++) {
                    for (int r = 0; r <= 1; r++) {
                        bool hasBoundary = b > 0;
                        bool useRandomNumbers = r > 0;
//                        std::cout << "========================> len=" << len << " offset=" << offset << " hasBoundary=" << hasBoundary << " useRandomNumbers=" << useRandomNumbers << std::endl;

                        auto t = generateInputAndExpected<T>(len, offset, hasBoundary, useRandomNumbers);
                        auto input = std::get<0>(t);
                        auto expected = std::get<1>(t);
                        PixelData<T> m(len, 1, 1, 0);
                        initFromZYXarray(m, input.data());
                        PixelData<T> expectedMesh(len, 1, 1, 0);
                        initFromZYXarray(expectedMesh, expected.data());

                        APRTimer timer(false);
                        LocalIntensityScale lis;

                        // Run on CPU old-impl
                        timer.start_timer("CPU Y-DIR");
                        PixelData<T> mCpu(m, true);
                        lis.calc_sat_mean_y(mCpu, offset, hasBoundary);
                        timer.stop_timer();

                        // Run on GPU
                        PixelData<T> mGpu(m, true);
                        timer.start_timer("GPU Y-DIR");
                        calcMean(mGpu, offset, MEAN_Y_DIR, hasBoundary);
                        timer.stop_timer();

                        // expectedMesh because of different order of calculation will have small floating-point differences
                        // comparing to CPU or GPU fast implementation, anyway GPU and CPU should have exactly same values!
                        EXPECT_EQ(compareMeshes(expectedMesh, mGpu, 0.00001), 0) << "---!!!!!!--- GPU values does not match";
                        EXPECT_EQ(compareMeshes(expectedMesh, mCpu, 0.00001), 0) << "---!!!!!!--- CPU values does not match";
                        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0) << "---!!!!!!--- CPU vs GPU values does not match";
                    }
                }
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_ALL_DIRS) {
        APRTimer timer(false);
        PixelData<float> m = getRandInitializedMesh<float>(33, 32, 31);

        LocalIntensityScale lis;
        for (int boundary = 0; boundary <= 1; boundary++) {
            for (int offset = 0; offset <= 6; ++offset) {
                bool hasBoundary = (boundary > 0);
//                std::cout << "========================> " << " offset=" << offset << " hasBoundary=" << hasBoundary << std::endl;

                // Run on CPU
                PixelData<float> mCpu(m, true);
                timer.start_timer("CPU mean ALL-DIR");
                lis.calc_sat_mean_y(mCpu, offset, hasBoundary);
                lis.calc_sat_mean_x(mCpu, offset, hasBoundary);
                lis.calc_sat_mean_z(mCpu, offset, hasBoundary);
                timer.stop_timer();

                // Run on GPU
                PixelData<float> mGpu(m, true);
                timer.start_timer("GPU mean ALL-DIR");
                calcMean(mGpu, offset, MEAN_ALL_DIR, hasBoundary);
                timer.stop_timer();

                // Compare results
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
            }
        }
    }

    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_ALL_DIRS_UINT16) {
        APRTimer timer(false);
        PixelData<uint16_t> m = getRandInitializedMesh<uint16_t>(33, 31, 13);

        LocalIntensityScale lis;
        for (int boundary = 0; boundary <= 1; boundary++) {
            for (int offset = 0; offset <= 6; ++offset) {
                bool hasBoundary = (boundary > 0);
//                std::cout << "========================> " << " offset=" << offset << " hasBoundary=" << hasBoundary << std::endl;

                // Run on CPU
                PixelData<uint16_t> mCpu(m, true);
                timer.start_timer("CPU mean ALL-DIR");
                lis.calc_sat_mean_y(mCpu, offset, hasBoundary);
                lis.calc_sat_mean_x(mCpu, offset, hasBoundary);
                lis.calc_sat_mean_z(mCpu, offset, hasBoundary);
                timer.stop_timer();

                // Run on GPU
                PixelData<uint16_t> mGpu(m, true);
                timer.start_timer("GPU mean ALL-DIR");
                calcMean(mGpu, offset, MEAN_ALL_DIR, hasBoundary);
                timer.stop_timer();

                // Compare results
                EXPECT_EQ(compareMeshes(mCpu, mGpu, 0), 0);
            }
        }
    }


    // ------------------------------------------------------------------------
    // Below tests are not yet fixed.
    // ------------------------------------------------------------------------


//    TEST(LocalIntensityScaleCudaTest, GPU_VS_CPU_FULL_PIPELINE) {
//        APRTimer timer(true);
//        PixelData<float> m = getRandInitializedMesh<float>(5, 5, 1, 25, 10, true);
//
//        APRParameters params;
//        params.sigma_th = 1;
//        params.sigma_th_max = 2;
//        params.reflect_bc_lis = true; //#TODO: @KG: The CPU pipeline uses this to true, so needs to now be implimented.
//
//        // Run on CPU
//        PixelData<float> mCpu(m, true);
//        PixelData<float> mCpuTemp(m, false);
//        timer.start_timer("CPU LIS FULL");
//
//        LocalIntensityScale localIntensityScale;
//
//        localIntensityScale.get_local_intensity_scale(mCpu, mCpuTemp, params);
//        timer.stop_timer();
//
//        // Run on GPU
//        PixelData<float> mGpu(m, true);
//        PixelData<float> mGpuTemp(m, false);
//        timer.start_timer("GPU LIS ALL-DIR");
//        getLocalIntensityScale(mGpu, mGpuTemp, params);
//        timer.stop_timer();
//
//        m.printMeshT(1);
//        mCpu.printMeshT(1);
//        mGpu.printMeshT(1);
//
//        // Compare results
//        //EXPECT_EQ(compareMeshes(mCpuTemp, mGpuTemp, 0.01), 0); //this is not needed these values are not required.
//        EXPECT_EQ(compareMeshes(mCpu, mGpu, 0.0000), 0);
//
//
//        PixelData<float> padd;
//        paddPixels(m, padd, 2, 2, 0);
//        m.printMeshT(1);
//        padd.printMeshT(1);
//    }


#endif // APR_USE_CUDA
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
