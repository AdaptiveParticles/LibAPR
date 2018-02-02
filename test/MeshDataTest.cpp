/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <gtest/gtest.h>
#include "src/data_structures/Mesh/MeshData.hpp"

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
        MeshData<MESH_TYPE> m{yLen, xLen, zLen};
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
        MeshData<MESH_TYPE> m{yLen, xLen, zLen};
    };

    TEST(MeshDataSimpleTest, ConstructorTest) {
        // default
        {
            MeshData<int> md;
            ASSERT_EQ(md.x_num, 0);
            ASSERT_EQ(md.y_num, 0);
            ASSERT_EQ(md.z_num, 0);
            ASSERT_EQ(md.mesh.size(), 0);
        }

        // size provided
        {
            MeshData<int> md(100, 200, 300);
            ASSERT_EQ(md.x_num, 200);
            ASSERT_EQ(md.y_num, 100);
            ASSERT_EQ(md.z_num, 300);
            ASSERT_EQ(md.mesh.size(), 100*200*300);
        }

        // mesh provided
        {
            // generate some data
            MeshData<int> md(3,4,5);
            for (size_t i = 0; i < md.mesh.size(); ++i) md.mesh[i] = i + 1;

            // test constructor
            MeshData<char> testedMesh(md, true);
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
        MeshData<NEW_TYPE> mf = m.to_type<NEW_TYPE>();
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
        ASSERT_EQ(m.access_no_protection(1, 2, 3), valueForIndex(1, 2, 3));

        // write test
        m(1, 2, 3) = 100;
        ASSERT_EQ(m(1, 2, 3), 100);
        m.access_no_protection(1, 2, 3) = 120;
        ASSERT_EQ(m(1, 2, 3), 120);

        // safe access beyond mesh size -> it should return last element
        ASSERT_EQ(m(yLen, xLen, zLen), valueForIndex(yLen-1, xLen-1, zLen-1));
    }

    TEST_P(MeshDataParameterTest, BlockCopyDataTest) {
        MeshData<unsigned short> mNew(yLen, xLen, zLen);

        int numOfBlocks = GetParam();
        mNew.block_copy_data(m, numOfBlocks);

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
    INSTANTIATE_TEST_CASE_P(CopyData, MeshDataParameterTest, ::testing::Values<int>(6, 7, 10000));

    TEST_F(MeshDataTest, InitializeTest) {
        {   // Size and initial value known
            MeshData<int> md;
            md.initialize(3, 4, 5, 123);
            ASSERT_EQ(md.y_num, 3);
            ASSERT_EQ(md.x_num, 4);
            ASSERT_EQ(md.z_num, 5);
            int size = 3 * 4 * 5;
            ASSERT_EQ(md.mesh.size(), size);
            for (int i = 0; i < size; ++i) ASSERT_EQ(md.mesh[i], 123);
        }
        {   // Use data from other mesh
            MeshData<int> md;
            md.initialize(m);
            ASSERT_EQ(md.x_num, xLen);
            ASSERT_EQ(md.y_num, yLen);
            ASSERT_EQ(md.z_num, zLen);
            ASSERT_EQ(md.mesh.size(), sizeOfMesh);
            for (int i = 0; i < sizeOfMesh; ++i) ASSERT_EQ(md.mesh[i], 0);
        }
        {   // Size and default value for type used
            MeshData<int> md;
            md.initialize(3, 4, 5);
            ASSERT_EQ(md.y_num, 3);
            ASSERT_EQ(md.x_num, 4);
            ASSERT_EQ(md.z_num, 5);
            int size = 3 * 4 * 5;
            ASSERT_EQ(md.mesh.size(), size);
        }
    }

    TEST_F(MeshDataTest, PreallocateTest) {
        {
            MeshData<int> md;
            md.preallocate(3, 5, 7);
            ASSERT_EQ(md.y_num, 2);
            ASSERT_EQ(md.x_num, 3);
            ASSERT_EQ(md.z_num, 4);
            int size = 2 * 3 * 4;
            ASSERT_EQ(md.mesh.size(), size);
        }
        {
            MeshData<int> md;
            md.preallocate(4, 6, 8);
            ASSERT_EQ(md.y_num, 2);
            ASSERT_EQ(md.x_num, 3);
            ASSERT_EQ(md.z_num, 4);
            int size = 2 * 3 * 4;
            ASSERT_EQ(md.mesh.size(), size);
        }
    }

    TEST(MeshDataSimpleTest, UnaryOpTest) {
        MeshData<int> m(1, 5, 1, 1);
        for (int i = 0; i < m.mesh.size(); ++i) m.mesh[i] = i + 1;

        MeshData<int> m2(1, 5, 1);
        m2.initWithUnaryOp(m, [](const int &a) { return a + 5; });

        for (int i = 0; i < m.mesh.size(); ++i) {
            ASSERT_EQ(m2.mesh[i], i + 1 + 5);
        }
    }

    TEST(MeshDataSimpleTest, DownSample) {
        {   // reduce/constant_operator calculate maximum value when downsampling
            MeshData<int> m(5, 6, 4);
            for (int i = 0; i < m.mesh.size(); ++i) m.mesh[i] = i + 1;

            MeshData<int> m2;
            down_sample(m, m2,
                        [](float x, float y) { return std::max(x, y); },
                        [](float x) { return x; },
                        true);
            int expected[] = {37, 39, 40, 47, 49, 50, 57, 59, 60, 97, 99, 100, 107, 109, 110, 117, 119, 120};
            for (int i = 0; i < m2.mesh.size(); ++i) {
                ASSERT_EQ(m2.mesh[i], expected[i]);
            }
        }
        {   // reduce/constant_operator calculate maximum value when downsampling
            MeshData<int> m(5, 6, 3);
            for (int i = 0; i < m.mesh.size(); ++i) m.mesh[i] = 5 * 6 * 3 - i;

            MeshData<int> m2;
            down_sample(m, m2,
                        [](float x, float y) { return std::max(x, y); },
                        [](float x) { return x; },
                        true);

            int expected[] = {90, 88, 86, 80, 78, 76, 70, 68, 66, 30, 28, 26, 20, 18, 16, 10, 8, 6};
            for (int i = 0; i < m2.mesh.size(); ++i) {
                ASSERT_EQ(m2.mesh[i], expected[i]);
            }
        }
        {
            // reduce/constant_operator calculate average value of pixels when downsampling
            MeshData<uint16_t> m(2, 2, 2);
            for (int i = 0; i < m.mesh.size(); ++i) m.mesh[i] = 8 - i;

            MeshData<float> m2;
            down_sample(m, m2,
                        [](const float &x, const float &y) -> float {return x + y; },
                        [](const float &x) -> float { return x/8.0; },
                        true);
            float expected[] = {4.5}; // (1+2+3+4+5+6+7+8)/8
            for (int i = 0; i < m2.mesh.size(); ++i) {
                ASSERT_EQ(m2.mesh[i], expected[i]);
            }
        }
    }

    TEST(MeshDataSimpleTest, DownSamplePyramid) {
        MeshData<float> m(4, 4, 4);
        for (int i = 0; i < m.mesh.size(); ++i) m.mesh[i] = i + 1;

        std::vector<MeshData<float>> ds;
        downsample_pyrmaid(m, ds, 3, 1);

        ASSERT_EQ(ds.size(), 4);
        ASSERT_EQ(ds[3].mesh.size(), 4 * 4 * 4); // original input
        ASSERT_EQ(ds[2].mesh.size(), 2 * 2 * 2);
        ASSERT_EQ(ds[1].mesh.size(), 1 * 1 * 1);
        ASSERT_EQ(ds[0].mesh.size(), 0); // not used - initialized by default constructor

        // check first and lass mesh
        for (int i = 0; i < ds[3].mesh.size(); ++i) ASSERT_EQ(ds[3].mesh[i], i + 1); // original
        ASSERT_EQ(ds[1].mesh[0], 32.5); // last = sum (1 + ... + 64) / 64
    }

    TEST(MeshDataSimpleTest, GetIdx) {
        MeshData<int> m(5, 6, 3);

        ASSERT_STREQ(m.getIdx(0).c_str(), "(0, 0, 0)");
        ASSERT_STREQ(m.getIdx(60).c_str(), "(0, 0, 2)");
        ASSERT_STREQ(m.getIdx(87).c_str(), "(2, 5, 2)");
        ASSERT_STREQ(m.getIdx(89).c_str(), "(4, 5, 2)");

        // out of bounds
        ASSERT_STREQ(m.getIdx(90).c_str(), "(ErrIdx)");
        ASSERT_STREQ(m.getIdx(-1).c_str(), "(ErrIdx)");
    }
};


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
