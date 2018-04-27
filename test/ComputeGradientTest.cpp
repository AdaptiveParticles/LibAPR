/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include "data_structures/Mesh/MeshData.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
#include "algorithm/ComputeBsplineRecursiveFilterCuda.h"
#include "algorithm/ComputeInverseCubicBsplineCuda.h"
#include <random>
#include "algorithm/APRConverter.hpp"

namespace {
    /**
     * Compares mesh with provided data
     * @param mesh
     * @param data - data with [Z][Y][X] structure
     * @return true if same
     */
    template<typename T>
    bool compare(MeshData<T> &mesh, const float *data, const float epsilon) {
        size_t dataIdx = 0;
        for (size_t z = 0; z < mesh.z_num; ++z) {
            for (size_t y = 0; y < mesh.y_num; ++y) {
                for (size_t x = 0; x < mesh.x_num; ++x) {
                    bool v = std::abs(mesh(y, x, z) - data[dataIdx]) < epsilon;
                    if (v == false) {
                        std::cerr << "Mesh and expected data differ. First place at (Y, X, Z) = " << y << ", " << x
                                  << ", " << z << ") " << mesh(y, x, z) << " vs " << data[dataIdx] << std::endl;
                        return false;
                    }
                    ++dataIdx;
                }
            }
        }
        return true;
    }

    /**
     * Compares two meshes
     * @param expected
     * @param tested
     * @param maxNumOfErrPrinted - how many error values should be printed (-1 for all)
     * @return number of errors detected
     */
    template <typename T>
    int compareMeshes(const MeshData<T> &expected, const MeshData<T> &tested, double maxError = 0.0001, int maxNumOfErrPrinted = 3) {
        int cnt = 0;
        for (size_t i = 0; i < expected.mesh.size(); ++i) {
            if (std::abs(expected.mesh[i] - tested.mesh[i]) > maxError || std::isnan(expected.mesh[i]) ||
                std::isnan(tested.mesh[i])) {
                if (cnt < maxNumOfErrPrinted || maxNumOfErrPrinted == -1) {
                    std::cout << "ERROR expected vs tested mesh: " << expected.mesh[i] << " vs " << tested.mesh[i] << " IDX:" << tested.getStrIndex(i) << std::endl;
                }
                cnt++;
            }
        }
        std::cout << "Number of errors / all points: " << cnt << " / " << expected.mesh.size() << std::endl;
        return cnt;
    }

    /**
     * Generates mesh with provided dims with random values in range [0, 1] * multiplier
     * @param y
     * @param x
     * @param z
     * @param multiplier
     * @return
     */
    template <typename T>
    MeshData<T> getRandInitializedMesh(int y, int x, int z, float multiplier = 2.0f, bool useIdxNumbers = false) {
        MeshData<T> m(y, x, z);
        std::cout << "Mesh info: " << m << std::endl;
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < m.mesh.size(); ++i) {
            m.mesh[i] = useIdxNumbers ? i : dist(mt) * multiplier;
        }
        return m;
    }

    template<typename T>
    bool initFromZYXarray(MeshData<T> &mesh, const float *data) {
        size_t dataIdx = 0;
        for (size_t z = 0; z < mesh.z_num; ++z) {
            for (size_t y = 0; y < mesh.y_num; ++y) {
                for (size_t x = 0; x < mesh.x_num; ++x) {
                    mesh(y, x, z) = data[dataIdx];
                    ++dataIdx;
                }
            }
        }
        return true;
    }

    TEST(ComputeGradientTest, 2D_XY) {
        {   // Corner points
            MeshData<float> m(6, 6, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {1.41, 0, 4.24,
                              0, 0, 0,
                              2.82, 0, 5.65};
            // put values in corners
            m(0, 0, 0) = 2;
            m(5, 0, 0) = 4;
            m(0, 5, 0) = 6;
            m(5, 5, 0) = 8;
            MeshData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.05));
        }
        {   // In the middle
            MeshData<float> m(6, 6, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {1, 1, 0,
                              1, 0, 0,
                              0, 0, 0};
            // put values in corners
            m(1, 1, 0) = 2;
            MeshData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }
        {   // One pixel image 1x1x1
            MeshData<float> m(1, 1, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0};
            // put values in corners
            m(0, 0, 0) = 2;
            MeshData<float> grad;
            grad.initDownsampled(m, 0);
            ComputeGradient cg;
            cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
            ASSERT_TRUE(compare(grad, expect, 0.01));
        }

    }

    TEST(ComputeGradientTest, Corners3D) {
        MeshData<float> m(6, 6, 4, 0);
        // expect gradient is 3x3x2 X/Y/Z plane
        float expect[] = {1.73, 0, 5.19,
                          0, 0, 0,
                          3.46, 0, 6.92,

                          8.66, 0, 12.12,
                          0, 0, 0,
                          10.39, 0, 13.85};
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        m(0, 0, 3) = 10;
        m(5, 0, 3) = 12;
        m(0, 5, 3) = 14;
        m(5, 5, 3) = 16;

        MeshData<float> grad;
        grad.initDownsampled(m, 0);
        ComputeGradient cg;
        cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.05));
    }

    TEST(ComputeGradientTest, 2D_XY_BSPLINE_Y_DIR) {
        {   // values in corners and in middle
            MeshData<float> m(5, 7, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.58, 0.00, 0.00, 0.08, 0.00, 0.00, 1.71,
                              0.56, 0.00, 0.00, 0.11, 0.00, 0.00, 1.51,
                              0.63, 0.00, 0.00, 0.11, 0.00, 0.00, 1.42,
                              0.88, 0.00, 0.00, 0.00, 0.00, 0.00, 1.75,
                              1.17, 0.00, 0.00, 0.00, 0.00, 0.00, 2.34};
            // put values in corners
            m(2, 3, 0) = 1;
            m(0, 0, 0) = 2;
            m(4, 0, 0) = 4;
            m(0, 6, 0) = 6;
            m(4, 6, 0) = 8;

            // Calculate bspline on CPU
            MeshData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
        {   // single point set in the middle
            MeshData<float> m(9, 3, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0.00, 0.01, 0.00,
                              0.00, 0.05, 0.00,
                              0.00, 0.12, 0.00,
                              0.00, 0.22, 0.00,
                              0.00, 0.28, 0.00,
                              0.00, 0.19, 0.00,
                              0.00, 0.08, 0.00,
                              0.00, 0.00, 0.00,
                              0.00, 0.00, 0.00};
            // put values in corners
            m(4, 1, 0) = 1;

            // Calculate bspline on CPU
            MeshData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
        {   // two pixel image 1x2x1
            MeshData<float> m(2, 1, 1, 0);
            // expect gradient is 3x3 X/Y plane
            float expect[] = {0,
                              0};
            // put values in corners
            m(0, 0, 0) = 1;

            // Calculate bspline on CPU
            MeshData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Y) {
        using ImgType = float;

        ImgType init[] =   {1.00, 0.00, 0.00,
                            1.00, 0.00, 6.00,
                            0.00, 6.00, 0.00,
                            6.00, 0.00, 0.00};

        ImgType expect[] = {1.00, 0.00, 2.00,
                            0.83, 1.00, 4.00,
                            1.17, 4.00, 1.00,
                            4.00, 2.00, 0.00};

        MeshData<ImgType> m(4, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        ComputeGradient().calc_inv_bspline_y(m);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_X) {
        using ImgType = float;

        ImgType init[] =   {0.00, 6.00, 0.00,
                            1.00, 0.00, 0.00,
                            0.00, 0.00, 1.00};

        ImgType expect[] = {2.00, 4.00, 2.00,
                            0.67, 0.16, 0.00,
                            0.00, 0.16, 0.67};

        MeshData<ImgType> m(3, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        ComputeGradient().calc_inv_bspline_x(m);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    // ======================= CUDA =======================================
    // ======================= CUDA =======================================
    // ======================= CUDA =======================================

#ifdef APR_USE_CUDA

    TEST(ComputeGradientTest, 2D_XY_CUDA) {
        // Corner points
        MeshData<float> m(6, 6, 1, 0);
        // expect gradient is 3x3 X/Y plane
        float expect[] = {1.41, 0, 4.24,
                          0, 0, 0,
                          2.82, 0, 5.65};
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        MeshData<float> grad;
        grad.initDownsampled(m, 0);
        cudaDownsampledGradient(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.01));
    }

    TEST(ComputeGradientTest, Corners3D_CUDA) {
        MeshData<float> m(6, 6, 4, 0);
        // expect gradient is 3x3x2 X/Y/Z plane
        float expect[] = {1.73, 0, 5.19,
                          0, 0, 0,
                          3.46, 0, 6.92,

                          8.66, 0, 12.12,
                          0, 0, 0,
                          10.39, 0, 13.85};
        // put values in corners
        m(0, 0, 0) = 2;
        m(5, 0, 0) = 4;
        m(0, 5, 0) = 6;
        m(5, 5, 0) = 8;
        m(0, 0, 3) = 10;
        m(5, 0, 3) = 12;
        m(0, 5, 3) = 14;
        m(5, 5, 3) = 16;

        MeshData<float> grad;
        grad.initDownsampled(m, 0);
        cudaDownsampledGradient(m, grad, 1, 1, 1);
        ASSERT_TRUE(compare(grad, expect, 0.01));
    }

    TEST(ComputeGradientTest, GPU_VS_CPU_ON_RANDOM_VALUES) {
        // Generate random mesh
        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(33, 31, 3);

        APRTimer timer(true);

        // Calculate gradient on CPU
        MeshData<ImgType> grad;
        grad.initDownsampled(m, 0);
        timer.start_timer("CPU gradient");
        ComputeGradient().calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
        timer.stop_timer();

        // Calculate gradient on GPU
        MeshData<ImgType> gradCuda;
        gradCuda.initDownsampled(m, 0);
        timer.start_timer("GPU gradient");
        cudaDownsampledGradient(m, gradCuda, 1, 1, 1);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(grad, gradCuda), 0);
    }

    TEST(ComputeBspineTest, BSPLINE_Y_DIR_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(129,127,128);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001;

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().bspline_filt_rec_y(mCpu, lambda, tolerance);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_Y_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeBspineTest, BSPLINE_X_DIR_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(129,127,128);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001;

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().bspline_filt_rec_x(mCpu, lambda, tolerance);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_X_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeBspineTest, BSPLINE_Z_DIR_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(129,127,128);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001;

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().bspline_filt_rec_z(mCpu, lambda, tolerance);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_Z_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeBspineTest, BSPLINE_FULL_XYZ_DIR_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(127, 128, 129);

        // Filter parameters
        const float lambda = 3;
        const float tolerance = 0.0001; // as defined in get_smooth_bspline_3D

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(m, true);
        timer.start_timer("CPU bspline");
        ComputeGradient().get_smooth_bspline_3D(mCpu, lambda);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(m, true);
        timer.start_timer("GPU bspline");
        cudaFilterBsplineFull(mGpu, lambda, tolerance, BSPLINE_ALL_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Y_CUDA) {
        using ImgType = float;

        ImgType init[] =   {1.00, 0.00, 0.00,
                            1.00, 0.00, 6.00,
                            0.00, 6.00, 0.00,
                            6.00, 0.00, 0.00};

        ImgType expect[] = {1.00, 0.00, 2.00,
                            0.83, 1.00, 4.00,
                            1.17, 4.00, 1.00,
                            4.00, 2.00, 0.00};

        MeshData<ImgType> m(4, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        m.printMesh(4,2);
        cudaInverseBspline(m, INV_BSPLINE_Y_DIR);
        m.printMesh(4,2);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Y_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(127, 33, 31);

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_y(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_Y_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_X_CUDA) {
        using ImgType = float;

        ImgType init[] =   {0.00, 6.00, 0.00,
                            1.00, 0.00, 0.00,
                            0.00, 0.00, 1.00};

        ImgType expect[] = {2.00, 4.00, 2.00,
                            0.67, 0.16, 0.00,
                            0.00, 0.16, 0.67};

        MeshData<ImgType> m(3, 3, 1);
        initFromZYXarray(m, init);

        // Calculate and compare
        m.printMesh(4,2);
        cudaInverseBspline(m, INV_BSPLINE_X_DIR);
        m.printMesh(4,2);
        ASSERT_TRUE(compare(m, expect, 0.01));
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_X_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(127, 61, 66);

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_x(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_X_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_Z_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(127, 61, 66);

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_z(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_Z_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeInverseBspline, CALC_INV_BSPLINE_FULL_XYZ_DIR_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(3,3,3,100);

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(m, true);
        timer.start_timer("CPU inv bspline");
        ComputeGradient().calc_inv_bspline_y(mCpu);
        ComputeGradient().calc_inv_bspline_x(mCpu);
        ComputeGradient().calc_inv_bspline_z(mCpu);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(m, true);
        timer.start_timer("GPU inv bspline");
        cudaInverseBspline(mGpu,  INV_BSPLINE_ALL_DIR);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeThreshold, CALC_THRESHOLD_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> m = getRandInitializedMesh<ImgType>(31, 33, 13);
        MeshData<ImgType> g = getRandInitializedMesh<ImgType>(31, 33, 13);
        float thresholdLevel = 1;

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(g, true);
        timer.start_timer("CPU threshold");
        ComputeGradient().threshold_gradient(mCpu, m, thresholdLevel);

        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(g, true);
        timer.start_timer("GPU threshold");
        thresholdGradient(mGpu, m, thresholdLevel);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeThreshold, CALC_THRESHOLD_IMG_RND_CUDA) {
        APRTimer timer(true);

        // Generate random mesh
        using ImgType = float;
        MeshData<ImgType> g = getRandInitializedMesh<ImgType>(31, 33, 13, 1, true);

        float thresholdLevel = 10;

        // Calculate bspline on CPU
        MeshData<ImgType> mCpu(g, true);
        timer.start_timer("CPU threshold");
        for (size_t i = 0; i < mCpu.mesh.size(); ++i) {
            if (mCpu.mesh[i] <= (thresholdLevel)) { mCpu.mesh[i] = thresholdLevel; }
        }
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImgType> mGpu(g, true);
        timer.start_timer("GPU threshold");
        thresholdImg(mGpu, thresholdLevel);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }

    TEST(ComputeThreshold, FULL_GRADIENT_TEST) {
        APRTimer timer(true);

        // Generate random mesh
        using ImageType = float;
        MeshData<ImageType> input_image = getRandInitializedMesh<ImageType>(310, 330, 13, 25);
        MeshData<ImageType> &image_temp = input_image;

        MeshData<ImageType> grad_temp; // should be a down-sampled image
        grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        MeshData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
        MeshData<float> local_scale_temp2;
        local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);

        MeshData<ImageType> grad_temp_GPU; // should be a down-sampled image
        grad_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        MeshData<float> local_scale_temp_GPU; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
        MeshData<float> local_scale_temp2_GPU;
        local_scale_temp2_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);


        APRParameters par;
        par.lambda = 3;
        par.Ip_th = 10;
        par.dx = 1;
        par.dy = 1;
        par.dz = 1;

        // Calculate bspline on CPU
        MeshData<ImageType> mCpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> CPU gradient");
        APRConverter<float>().get_gradient(mCpuImage, grad_temp, local_scale_temp, local_scale_temp2, 0, par);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImageType> mGpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> GPU gradient");
        getGradient(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpuImage, mGpuImage), 0);
        EXPECT_EQ(compareMeshes(grad_temp, grad_temp_GPU), 0);
        EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU), 0);
    }

    TEST(ComputeThreshold, FULL_PIPELINE_TEST) {
        APRTimer timer(true);

        // Generate random mesh
        using ImageType = float;
        MeshData<ImageType> input_image = getRandInitializedMesh<ImageType>(310, 330, 32, 25);
        MeshData<ImageType> &image_temp = input_image;

        MeshData<ImageType> grad_temp; // should be a down-sampled image
        grad_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        MeshData<float> local_scale_temp; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
        MeshData<float> local_scale_temp2;
        local_scale_temp2.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);

        MeshData<ImageType> grad_temp_GPU; // should be a down-sampled image
        grad_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num, 0);
        MeshData<float> local_scale_temp_GPU; // Used as down-sampled images for some averaging steps where it is useful to not lose precision, or get over-flow errors
        local_scale_temp_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);
        MeshData<float> local_scale_temp2_GPU;
        local_scale_temp2_GPU.initDownsampled(input_image.y_num, input_image.x_num, input_image.z_num);


        APRParameters par;
        par.lambda = 3;
        par.Ip_th = 10;
        par.sigma_th = 0;
        par.sigma_th_max = 0;
        par.dx = 1;
        par.dy = 1;
        par.dz = 1;

        // Calculate bspline on CPU
        MeshData<ImageType> mCpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> CPU PIPELINE");
        APRConverter<float>().get_gradient(mCpuImage, grad_temp, local_scale_temp, local_scale_temp2, 0, par);
        APRConverter<float>().get_local_intensity_scale(local_scale_temp, local_scale_temp2, par);
        timer.stop_timer();

        // Calculate bspline on GPU
        MeshData<ImageType> mGpuImage(image_temp, true);
        timer.start_timer(">>>>>>>>>>>>>>>>> GPU PIPELINE");
        getFullPipeline(mGpuImage, grad_temp_GPU, local_scale_temp_GPU, local_scale_temp2_GPU, 0, par);
        timer.stop_timer();

        // Compare GPU vs CPU
        EXPECT_EQ(compareMeshes(mCpuImage, mGpuImage), 0);
        EXPECT_EQ(compareMeshes(grad_temp, grad_temp_GPU), 0);
        EXPECT_EQ(compareMeshes(local_scale_temp2, local_scale_temp2_GPU, 0.01), 0);
        EXPECT_EQ(compareMeshes(local_scale_temp, local_scale_temp_GPU, 0.01), 0);
    }


#endif // APR_USE_CUDA

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
