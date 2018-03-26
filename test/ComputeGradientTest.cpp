/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <array>
#include <gtest/gtest.h>
#include "data_structures/Mesh/MeshData.hpp"
#include "algorithm/ComputeGradient.hpp"
#include "algorithm/ComputeGradientCuda.hpp"
//#include "algorithm/ComputeGradientCudaRegs.h"
#include <random>

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
            ASSERT_TRUE(compare(grad, expect, 0.01));
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
        ASSERT_TRUE(compare(grad, expect, 0.01));
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
            float expect[] = {0.41,
                              0.44 };
            // put values in corners
            m(0, 0, 0) = 1;

            // Calculate bspline on CPU
            MeshData<float> mCpu(m, true);
            ComputeGradient cg;
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            ASSERT_TRUE(compare(mCpu, expect, 0.01));
        }
    }

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

    TEST(ComputeGradientTest, RandomInputCompareToCpuVersion_CUDA) {
        // Generate random mesh
        MeshData<float> m(127, 128, 129, 0);
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < m.mesh.size(); ++i) {
            m.mesh[i] = dist(mt);
        }

        APRTimer timer;
        timer.verbose_flag = true;

        // Calculate gradient on CPU
        MeshData<float> grad;
        grad.initDownsampled(m, 0);
        timer.start_timer("CPU gradient");
        ComputeGradient cg;
        cg.calc_bspline_fd_ds_mag(m, grad, 1, 1, 1);
        timer.stop_timer();

        // Calculate gradient on GPU
        MeshData<float> gradCuda;
        gradCuda.initDownsampled(m, 0);
        timer.start_timer("GPU gradient");
        cudaDownsampledGradient(m, gradCuda, 1, 1, 1);
        timer.stop_timer();

        // Compare GPU vs CPU
        bool once = true;
        int cnt = 0;
        for (size_t i = 0; i < grad.mesh.size(); ++i) {
            if (std::abs(grad.mesh[i] - gradCuda.mesh[i]) > 0.0001) {
                if (once) {
                    std::cout << "ERR " << grad.mesh[i] << " vs " << gradCuda.mesh[i] << std::endl;
                    once = false;
                }
                cnt++;
            }
        }
        std::cout << "Number of errors / Number of gradient points: " << cnt << " / " << grad.mesh.size() << std::endl;
        EXPECT_EQ(cnt, 0);
    }


    TEST(ComputeGradientTest, 2D_XY_BSPLINE_Y_DIR_CUDA) {
        {
            APRTimer timer;
            timer.verbose_flag = false;

            MeshData<float> m(9, 3, 1, 0);
            // put value in the middle
            m(4, 1, 0) = 2;

            // Calculate bspline on CPU
            MeshData<float> mCpu(m, true);
            ComputeGradient cg;
            timer.start_timer("CPU y-dir spline");
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            timer.stop_timer();

            // Calculate bspline on GPU
            MeshData<float> mGpu(m, true);
            timer.start_timer("GPU y-dir spline");
            cudaFilterBsplineYdirection(mGpu, 3.0, 0.0001);
            timer.stop_timer();

            // Compare GPU vs CPU
            bool once = true;
            int cnt = 0;
            for (size_t i = 0; i < mCpu.mesh.size(); ++i) {
                if (std::abs(mCpu.mesh[i] - mGpu.mesh[i]) > 0.0001) {
                    if (once) {
                        std::cout << "ERR " << mCpu.mesh[i] << " vs " << mGpu.mesh[i] << std::endl;
                        once = false;
                    }
                    cnt++;
                }
            }
            std::cout << "Number of errors / Number of gradient points: " << cnt << " / " << mCpu.mesh.size() << std::endl;
            EXPECT_EQ(cnt, 0);
        }
        {
            APRTimer timer;
            timer.verbose_flag = false;

            // Generate random mesh
            MeshData<float> m(128, 512, 512);
            std::cout << m << std::endl;
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < m.mesh.size(); ++i) {
                m.mesh[i] = dist(mt);
            }

            // Calculate bspline on CPU
            MeshData<float> mCpu(m, true);
            ComputeGradient cg;
            timer.start_timer("CPU y-dir spline");
            cg.bspline_filt_rec_y(mCpu, 3.0, 0.0001);
            timer.stop_timer();

            // Calculate bspline on GPU
            MeshData<float> mGpu(m, true);
            timer.start_timer("GPU y-dir spline");
            cudaFilterBsplineYdirection(mGpu, 3.0, 0.0001);
            timer.stop_timer();

            // Compare GPU vs CPU
            bool once = true;
            int cnt = 0;
            for (size_t i = 0; i < mCpu.mesh.size(); ++i) {
                if (std::abs(mCpu.mesh[i] - mGpu.mesh[i]) > 0.0001) {
                    if (once) {
                        std::cout << "ERR " << mCpu.mesh[i] << " vs " << mGpu.mesh[i] << std::endl;
                        once = false;
                    }
                    cnt++;
                }
            }
            std::cout << "Number of errors / Number of gradient points: " << cnt << " / " << mCpu.mesh.size()
                      << std::endl;
            EXPECT_EQ(cnt, 0);
        }
    }

    TEST(ComputeGradientTest, 2D_XY_BSPLINE_Y_DIR_CUDA_NEW) {
        {
            std::cout << "\n---------------------------------\n\n";
            APRTimer timer;
            timer.verbose_flag = true;

            // Generate random mesh
            using ImgType = float ;
            MeshData<ImgType> m(129, 129, 512);
            std::cout << m << std::endl;
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < m.mesh.size(); ++i) {
                m.mesh[i] = dist(mt) * 2;
            }

            const float lambda = 3;
            const float tolerance = 0.001;

            // Calculate bspline on CPU
            MeshData<ImgType> mCpu(m, true);
            ComputeGradient cg;
            timer.start_timer("CPU y-dir spline ======================================================================================== ");
            cg.bspline_filt_rec_y(mCpu, lambda, tolerance);
            timer.stop_timer();

            // Calculate bspline on GPU
            MeshData<ImgType> mGpu(m, true);
            timer.start_timer("GPU y-dir spline");
            cudaFilterBsplineYdirection(mGpu, lambda, tolerance);
            timer.stop_timer();

            // Compare GPU vs CPU
            int cnt = 0;
            for (size_t i = 0; i < mCpu.mesh.size(); ++i) {
                if (std::abs(mCpu.mesh[i] - mGpu.mesh[i]) > 0.0001) {
                    if (cnt < 3) {
                        std::cout << "ERR " << mCpu.mesh[i] << " vs " << mGpu.mesh[i] << " IDX:" << mGpu.getStrIndex(i) << std::endl;
                    }
                    cnt++;
                }
            }
            std::cout << "Number of errors / Number of gradient points: " << cnt << " / " << mCpu.mesh.size() << std::endl;
            EXPECT_EQ(cnt, 0);
        }
    }

    TEST(ComputeGradientTest, REGISTERS_USEX) {
        {
            std::cout << "\n---------------------------------\n\n";
            APRTimer timer;
            timer.verbose_flag = true;

            // Generate random mesh
            bool show = false;
            using ImgType = float ;
            MeshData<ImgType> m(129, 127, 1024);
            std::cout << m << std::endl;
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < m.mesh.size(); ++i) {
                m.mesh[i] = dist(mt) * 2;
            }

            const float lambda = 3;
            const float tolerance = 0.001;

            // Calculate bspline on CPU
            MeshData<ImgType> mCpu(m, true);
            ComputeGradient cg;
            timer.start_timer("CPU y-dir spline ======================================================================================== ");
            cg.bspline_filt_rec_x(mCpu, lambda, tolerance);
            timer.stop_timer();

            // Calculate bspline on GPU
            MeshData<ImgType> mGpu(m, true);
            timer.start_timer("GPU y-dir spline");
            cudaFilterBsplineXdirection(mGpu, lambda, tolerance);
            timer.stop_timer();

            if (show) {
                m.printMesh(5, 1);
                mCpu.printMesh(5, 1);
                mGpu.printMesh(5, 1);
            }
            // Compare GPU vs CPU
            int cnt = 0;
            for (size_t i = 0; i < mCpu.mesh.size(); ++i) {
                if (std::abs(mCpu.mesh[i] - mGpu.mesh[i]) > 0.0001) {
                    if (cnt < 3) {
                        std::cout << "ERR " << mCpu.mesh[i] << " vs " << mGpu.mesh[i] << " IDX:" << mGpu.getStrIndex(i) << std::endl;
                    }
                    cnt++;
                }
            }
            std::cout << "Number of errors / Number of gradient points: " << cnt << " / " << mCpu.mesh.size() << std::endl;
            EXPECT_EQ(cnt, 0);
        }
    }

    TEST(ComputeGradientTest, REGISTERS_USEZ) {
        {
            std::cout << "\n---------------------------------\n\n";
            APRTimer timer;
            timer.verbose_flag = true;

            // Generate random mesh
            bool show = false;
            using ImgType = float ;
            MeshData<ImgType> m(129, 127, 1024);
            std::cout << m << std::endl;
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (size_t i = 0; i < m.mesh.size(); ++i) {
                m.mesh[i] = dist(mt) * 2;
            }

            const float lambda = 3;
            const float tolerance = 0.001;

            // Calculate bspline on CPU
            MeshData<ImgType> mCpu(m, true);
            ComputeGradient cg;
            timer.start_timer("CPU y-dir spline ======================================================================================== ");
            cg.bspline_filt_rec_z(mCpu, lambda, tolerance);
            timer.stop_timer();

            // Calculate bspline on GPU
            MeshData<ImgType> mGpu(m, true);
            timer.start_timer("GPU y-dir spline");
            cudaFilterBsplineZdirection(mGpu, lambda, tolerance);
            timer.stop_timer();

            if (show) {
                m.printMesh(5, 1);
                mCpu.printMesh(5, 1);
                mGpu.printMesh(5, 1);
            }
            // Compare GPU vs CPU
            int cnt = 0;
            for (size_t i = 0; i < mCpu.mesh.size(); ++i) {
                if (std::abs(mCpu.mesh[i] - mGpu.mesh[i]) > 0.0001) {
                    if (cnt < 3) {
                        std::cout << "ERR " << mCpu.mesh[i] << " vs " << mGpu.mesh[i] << " IDX:" << mGpu.getStrIndex(i) << std::endl;
                    }
                    cnt++;
                }
            }
            std::cout << "Number of errors / Number of gradient points: " << cnt << " / " << mCpu.mesh.size() << std::endl;
            EXPECT_EQ(cnt, 0);
        }
    }

#endif // APR_USE_CUDA

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
