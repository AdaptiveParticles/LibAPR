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
            md.preallocate(3, 5, 7, 123);
            ASSERT_EQ(md.y_num, 2);
            ASSERT_EQ(md.x_num, 3);
            ASSERT_EQ(md.z_num, 4);
            int size = 2 * 3 * 4;
            ASSERT_EQ(md.mesh.size(), size);
            for (int i = 0; i < size; ++i) ASSERT_EQ(md.mesh[i], 123);
        }
        {
            MeshData<int> md;
            md.preallocate(4, 6, 8, 13);
            ASSERT_EQ(md.y_num, 2);
            ASSERT_EQ(md.x_num, 3);
            ASSERT_EQ(md.z_num, 4);
            int size = 2 * 3 * 4;
            ASSERT_EQ(md.mesh.size(), size);
            for (int i = 0; i < size; ++i) ASSERT_EQ(md.mesh[i], 13);
        }
    }
};


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
