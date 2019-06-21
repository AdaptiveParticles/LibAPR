/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <gtest/gtest.h>
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/Mesh/PixelDataCuda.h"
#include <random>

namespace {
    class MeshDataTest : public ::testing::Test {
        typedef char MESH_TYPE;
    protected:
         void SetUp() override {
            // Initialize it with some data
            for (int y = 0; y < yLen; ++y) {
                for (int x = 0; x < xLen; ++x) {
                    for (int z = 0; z < zLen; ++z) {
                        m(y, x, z) = valueForIndex(y, x, z);
                    }
                }
            }
        }

        MESH_TYPE valueForIndex(int y, int x, int z) {return (size_t)z * xLen * yLen + x * yLen + y + 1;}

        const int yLen = 113;
        const int xLen = 254;
        const int zLen = 123;
        const size_t sizeOfMesh = (size_t)yLen * xLen * zLen;
        PixelData<MESH_TYPE> m{yLen, xLen, zLen};
    };

    class MeshDataParameterTest : public ::testing::TestWithParam<int> {
        typedef int MESH_TYPE;
    protected:
        void SetUp() override {
            // Initialize it with some data
            for (int y = 0; y < yLen; ++y) {
                for (int x = 0; x < xLen; ++x) {
                    for (int z = 0; z < zLen; ++z) {
                        m(y, x, z) = valueForIndex(y, x, z);
                    }
                }
            }
        }

        MESH_TYPE valueForIndex(int y, int x, int z) {return (size_t)z * xLen * yLen + x * yLen + y + 1;}

        const int yLen = 10;
        const int xLen = 20;
        const int zLen = 30;
        const size_t sizeOfMesh = (size_t)yLen * xLen * zLen;
        PixelData<MESH_TYPE> m{yLen, xLen, zLen};
    };

    TEST(MeshDataSimpleTest, ConstructorTest) {
        // default
        {
            PixelData<int> md;
            ASSERT_EQ(md.x_num, 0);
            ASSERT_EQ(md.y_num, 0);
            ASSERT_EQ(md.z_num, 0);
            ASSERT_EQ(md.mesh.size(), 0);
        }

        // size provided
        {
            PixelData<int> md(100, 200, 300);
            ASSERT_EQ(md.x_num, 200);
            ASSERT_EQ(md.y_num, 100);
            ASSERT_EQ(md.z_num, 300);
            ASSERT_EQ(md.mesh.size(), 100*200*300);
        }

        // mesh provided
        {
            // generate some data
            PixelData<int> md(3,4,5);
            for (size_t i = 0; i < md.mesh.size(); ++i) md.mesh[i] = i + 1;

            // test constructor
            PixelData<char> testedMesh(md, true);
            ASSERT_EQ(md.mesh.size(), testedMesh.mesh.size());
            ASSERT_EQ(md.x_num, testedMesh.x_num);
            ASSERT_EQ(md.y_num, testedMesh.y_num);
            ASSERT_EQ(md.z_num, testedMesh.z_num);
            for (size_t i = 0; i < md.mesh.size(); ++i) {
                ASSERT_EQ(testedMesh.mesh[i], (char)i + 1);
            }
        }
    }

    TEST_F(MeshDataTest, ToTypeTest) {
        // Change type and compare if still OK
        typedef short NEW_TYPE;
        PixelData<NEW_TYPE> mf = m.toType<NEW_TYPE>();
        ASSERT_EQ(mf.x_num, xLen);
        ASSERT_EQ(mf.y_num, yLen);
        ASSERT_EQ(mf.z_num, zLen);
        ASSERT_EQ(mf.mesh.size(), m.mesh.size());
        for (int y = 0; y < yLen; ++y) {
            for (int x = 0; x < xLen; ++x) {
                for (int z = 0; z < zLen; ++z) {
                    ASSERT_EQ(mf(y, x, z), static_cast<NEW_TYPE>(m(y, x, z)));
                }
            }
        }
    }

    TEST_F(MeshDataTest, AccessorsTest) {
        // read test
        ASSERT_EQ(m(1, 2, 3), valueForIndex(1, 2, 3)  /* (3 * 20 * 10) + (2 * 10) + (1) + 1 */);
        ASSERT_EQ(m.at(1, 2, 3), valueForIndex(1, 2, 3));

        // write test
        m(1, 2, 3) = 100;
        ASSERT_EQ(m(1, 2, 3), 100);
        m.at(1, 2, 3) = 120;
        ASSERT_EQ(m(1, 2, 3), 120);

        // safe access beyond mesh size -> it should return last element
        ASSERT_EQ(m(yLen, xLen, zLen), valueForIndex(yLen-1, xLen-1, zLen-1));
    }

    TEST_P(MeshDataParameterTest, BlockCopyDataTest) {
        PixelData<unsigned short> mNew(yLen, xLen, zLen);

        int numOfBlocks = GetParam();
        mNew.copyFromMesh(m, numOfBlocks);

        // Compare if same
        for (int y = 0; y < yLen; ++y) {
            for (int x = 0; x < xLen; ++x) {
                for (int z = 0; z < zLen; ++z) {
                    ASSERT_EQ(m(y, x, z), mNew(y, x, z));
                }
            }
        }
    }

    // Run with different number of blocks (easy/not easy dividable
    // and exceeding number of elements in mesh)
    INSTANTIATE_TEST_CASE_P(CopyData, MeshDataParameterTest, ::testing::Values<int>(6, 7, 10000),);

    TEST_F(MeshDataTest, InitializeTest) {
        {   // Size and initial value known
            PixelData<int> md;
            md.initWithValue(3, 4, 5, 123);
            ASSERT_EQ(md.y_num, 3);
            ASSERT_EQ(md.x_num, 4);
            ASSERT_EQ(md.z_num, 5);
            int size = 3 * 4 * 5;
            ASSERT_EQ(md.mesh.size(), size);
            for (int i = 0; i < size; ++i) ASSERT_EQ(md.mesh[i], 123);
        }
        {   // Use data from other mesh
            PixelData<int> md;
            md.init(m);
            ASSERT_EQ(md.x_num, xLen);
            ASSERT_EQ(md.y_num, yLen);
            ASSERT_EQ(md.z_num, zLen);
            ASSERT_EQ(md.mesh.size(), sizeOfMesh);
        }
        {   // Size and default value for type used
            PixelData<int> md;
            md.init(3, 4, 5);
            ASSERT_EQ(md.y_num, 3);
            ASSERT_EQ(md.x_num, 4);
            ASSERT_EQ(md.z_num, 5);
            int size = 3 * 4 * 5;
            ASSERT_EQ(md.mesh.size(), size);
        }
    }

    TEST_F(MeshDataTest, InitDownsampledTest) {
        {
            PixelData<int> md;
            md.initDownsampled(3, 5, 7, false);
            ASSERT_EQ(md.y_num, 2);
            ASSERT_EQ(md.x_num, 3);
            ASSERT_EQ(md.z_num, 4);
            int size = 2 * 3 * 4;
            ASSERT_EQ(md.mesh.size(), size);
        }
        {
            PixelData<int> md;
            md.initDownsampled(4, 6, 8, false);
            ASSERT_EQ(md.y_num, 2);
            ASSERT_EQ(md.x_num, 3);
            ASSERT_EQ(md.z_num, 4);
            int size = 2 * 3 * 4;
            ASSERT_EQ(md.mesh.size(), size);
        }
        {
            PixelData<int> md;
            md.initDownsampled(PixelData<char>(4, 6, 8));
            ASSERT_EQ(md.y_num, 2);
            ASSERT_EQ(md.x_num, 3);
            ASSERT_EQ(md.z_num, 4);
            int size = 2 * 3 * 4;
            ASSERT_EQ(md.mesh.size(), size);
        }
        {
            PixelData<int> md;
            md.initDownsampled(PixelData<float>(3, 5, 7), 2);
            ASSERT_EQ(md.y_num, 2);
            ASSERT_EQ(md.x_num, 3);
            ASSERT_EQ(md.z_num, 4);
            int size = 2 * 3 * 4;
            ASSERT_EQ(md.mesh.size(), size);
            for (size_t i = 0; i < md.mesh.size(); ++i) ASSERT_EQ(md.mesh[i], 2);
        }
    }

    TEST(MeshDataSimpleTest, UnaryOpTest) {
        PixelData<int> m(1, 5, 1, 1);
        for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = i + 1;

        PixelData<int> m2(1, 5, 1);
        m2.copyFromMeshWithUnaryOp(m, [](const int &a) { return a + 5; });

        for (size_t i = 0; i < m.mesh.size(); ++i) {
            ASSERT_EQ(m2.mesh[i], i + 1 + 5);
        }
    }

    TEST(MeshDataSimpleTest, DownSample) {
        {   // reduce/constant_operator calculate maximum value when downsampling
            PixelData<int> m(5, 6, 4);
            for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = i + 1;

            PixelData<int> m2;
            downsample(m, m2,
                       [](float x, float y) { return std::max(x, y); },
                       [](float x) { return x; },
                       true);
            int expected[] = {37, 39, 40, 47, 49, 50, 57, 59, 60, 97, 99, 100, 107, 109, 110, 117, 119, 120};
            for (size_t i = 0; i < m2.mesh.size(); ++i) {
                ASSERT_EQ(m2.mesh[i], expected[i]);
            }
        }
        {   // reduce/constant_operator calculate maximum value when downsampling
            PixelData<int> m(5, 6, 3);
            for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = 5 * 6 * 3 - i;

            PixelData<int> m2;
            downsample(m, m2,
                       [](float x, float y) { return std::max(x, y); },
                       [](float x) { return x; },
                       true);

            int expected[] = {90, 88, 86, 80, 78, 76, 70, 68, 66, 30, 28, 26, 20, 18, 16, 10, 8, 6};
            for (size_t i = 0; i < m2.mesh.size(); ++i) {
                ASSERT_EQ(m2.mesh[i], expected[i]);
            }
        }
        {
            // reduce/constant_operator calculate average value of pixels when downsampling
            PixelData<uint16_t> m(2, 2, 2);
            for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = 8 - i;

            PixelData<float> m2;
            downsample(m, m2,
                       [](const float &x, const float &y) -> float { return x + y; },
                       [](const float &x) -> float { return x / 8.0; },
                       true);
            float expected[] = {4.5}; // (1+2+3+4+5+6+7+8)/8
            for (size_t i = 0; i < m2.mesh.size(); ++i) {
                ASSERT_EQ(m2.mesh[i], expected[i]);
            }
        }
    }

    TEST(MeshDataSimpleTest, DownSamplePyramid) {
        PixelData<float> m(4, 4, 4);
        for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = i + 1;

        std::vector<PixelData<float>> ds;
        downsamplePyrmaid(m, ds, 3, 1);

        ASSERT_EQ(ds.size(), 4);
        ASSERT_EQ(ds[3].mesh.size(), 4 * 4 * 4); // original input
        ASSERT_EQ(ds[2].mesh.size(), 2 * 2 * 2);
        ASSERT_EQ(ds[1].mesh.size(), 1 * 1 * 1);
        ASSERT_EQ(ds[0].mesh.size(), 0); // not used - initialized by default constructor

        // check first and lass mesh
        for (size_t i = 0; i < ds[3].mesh.size(); ++i) ASSERT_EQ(ds[3].mesh[i], i + 1); // original
        ASSERT_EQ(ds[1].mesh[0], 32.5); // last = sum (1 + ... + 64) / 64
    }

    TEST(MeshDataSimpleTest, GetIdx) {
        PixelData<int> m(5, 6, 3);

        ASSERT_STREQ(m.getStrIndex(0).c_str(), "(0, 0, 0)");
        ASSERT_STREQ(m.getStrIndex(60).c_str(), "(0, 0, 2)");
        ASSERT_STREQ(m.getStrIndex(87).c_str(), "(2, 5, 2)");
        ASSERT_STREQ(m.getStrIndex(89).c_str(), "(4, 5, 2)");

        // out of bounds
        ASSERT_STREQ(m.getStrIndex(90).c_str(), "(ErrIdx)");
        ASSERT_STREQ(m.getStrIndex(-1).c_str(), "(ErrIdx)");
    }

    TEST(MeshDataSimpleTest, PadArray) {
        PixelData<int> m(2, 2, 2);




    }
}

#ifdef APR_USE_CUDA
namespace {
    /**
     * Compares two meshes
     * @param expected
     * @param tested
     * @param maxNumOfErrPrinted - how many error values should be printed (-1 for all)
     * @return number of errors detected
     */
    template <typename T>
    int compareMeshes(const PixelData<T> &expected, const PixelData<T> &tested, double maxError = 0.0001, int maxNumOfErrPrinted = 3) {
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
    PixelData<T> getRandInitializedMesh(int y, int x, int z, float multiplier = 2.0f, bool useIdxNumbers = false) {
        PixelData<T> m(y, x, z);
        std::cout << "Mesh info: " << m << std::endl;
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < m.mesh.size(); ++i) {
            m.mesh[i] = useIdxNumbers ? i : dist(mt) * multiplier;
        }
        return m;
    }
}
TEST(MeshDataSimpleTest, DownSampleCuda) {
    {   // reduce/constant_operator calculate maximum value when downsampling
        PixelData<float> m(5, 6, 4);
        for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = i + 1;

        PixelData<float> m2; m2.initDownsampled(m);
        downsampleMaxCuda(m, m2);
        int expected[] = {37, 39, 40, 47, 49, 50, 57, 59, 60, 97, 99, 100, 107, 109, 110, 117, 119, 120};
        for (size_t i = 0; i < m2.mesh.size(); ++i) {
            ASSERT_EQ(m2.mesh[i], expected[i]);
        }
    }
    {   // reduce/constant_operator calculate maximum value when downsampling
        PixelData<float> m(5, 6, 3);
        for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = 5 * 6 * 3 - i;

        PixelData<float> m2; m2.initDownsampled(m);
        downsampleMaxCuda(m, m2);

        int expected[] = {90, 88, 86, 80, 78, 76, 70, 68, 66, 30, 28, 26, 20, 18, 16, 10, 8, 6};
        for (size_t i = 0; i < m2.mesh.size(); ++i) {
            ASSERT_EQ(m2.mesh[i], expected[i]);
        }
    }
    {
        // reduce/constant_operator calculate average value of pixels when downsampling
        PixelData<float> m(2, 2, 2);
        for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = 8 - i;

        PixelData<float> m2; m2.initDownsampled(m);
        downsampleMeanCuda(m, m2);
        float expected[] = {4.5}; // (1+2+3+4+5+6+7+8)/8
        for (size_t i = 0; i < m2.mesh.size(); ++i) {
            ASSERT_EQ(m2.mesh[i], expected[i]);
        }
    }
    {
        // reduce/constant_operator calculate average value of pixels when downsampling
        PixelData<float> m(3, 3, 3);
        for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = 27 - i;

        PixelData<float> mCpu;
        downsample(m, mCpu,
                   [](const float &x, const float &y) -> float { return x + y; },
                   [](const float &x) -> float { return x / 8.0; },
                   true);

        PixelData<float> mGpu; mGpu.initDownsampled(m);
        downsampleMeanCuda(m, mGpu);

        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }
    {
        APRTimer timer(true);

        // reduce/constant_operator calculate average value of pixels when downsampling
        PixelData<float> m =  getRandInitializedMesh<float>(33, 22, 21);
        for (size_t i = 0; i < m.mesh.size(); ++i) m.mesh[i] = 27 - i;

        PixelData<float> mCpu; mCpu.initDownsampled(m);
        timer.start_timer("CPU downsample");
        downsample(m, mCpu,
                   [](const float &x, const float &y) -> float { return x + y; },
                   [](const float &x) -> float { return x / 8.0; },
                   false);
        timer.stop_timer();

        PixelData<float> mGpu; mGpu.initDownsampled(m);
        timer.start_timer("GPU downsample");
        downsampleMeanCuda(m, mGpu);
        timer.stop_timer();

        EXPECT_EQ(compareMeshes(mCpu, mGpu), 0);
    }
}
#endif

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
