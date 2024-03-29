/*
 * Created by Krzysztof Gonciarz 2018
 */
#include <gtest/gtest.h>
#include "data_structures/Mesh/PixelData.hpp"
#include "data_structures/Mesh/PixelDataCuda.h"
#include <random>

namespace {


    class VectorDataTest : public ::testing::Test {
        typedef int MESH_TYPE;
    protected:
        void SetUp() override {
            m.init(sz);
            // Initialize it with some data
            for (int y = 0; y < sz; ++y) {
                m[y] = y;
            }
        }
        const int sz = 113;

        VectorData<MESH_TYPE> m;
    };

    TEST(MeshDataSimpleTest, PixelDataDimTest) {
        // size provided
        {
            PixelData<int> md(10, 20, 30);
            auto d = md.getDimension();

            ASSERT_EQ(d.y, 10);
            ASSERT_EQ(d.x, 20);
            ASSERT_EQ(d.z, 30);
            ASSERT_EQ(d.size(), 10*20*30);
        }
        { // adding int to all dims

            PixelDataDim x = {1,2,3};
            auto d = x + 1;
            ASSERT_EQ(d.y, 2);
            ASSERT_EQ(d.x, 3);
            ASSERT_EQ(d.z, 4);
        }
        { // subtract int from all dims

            const PixelDataDim x = {1,2,3};
            auto d = x - 1;
            ASSERT_EQ(d.y, 0);
            ASSERT_EQ(d.x, 1);
            ASSERT_EQ(d.z, 2);
        }
        { // adding another PixelDataDim

            PixelDataDim x = {1,2,3};
            const PixelDataDim y = {5, 10, 15};
            auto d = x + y;
            ASSERT_EQ(d.y, 6);
            ASSERT_EQ(d.x, 12);
            ASSERT_EQ(d.z, 18);
        }
        { // subtract another PixelDataDim

            const PixelDataDim x = {5, 10, 15};
            PixelDataDim y = {1, 2, 3};
            auto d = x - y;
            ASSERT_EQ(d.y, 4);
            ASSERT_EQ(d.x, 8);
            ASSERT_EQ(d.z, 12);
        }
        { // compare two PixelDataDim structures
            const PixelDataDim x = {2, 3, 5};
            const PixelDataDim y = {2, 3, 5};
            const PixelDataDim z = {3, 4, 5};

            ASSERT_TRUE(x == y);
            ASSERT_FALSE(x != y);

            ASSERT_FALSE(x == z);
            ASSERT_TRUE(x != z);
        }
    }

    TEST_F(VectorDataTest, InitTest) {
        // Check initialize and resize and size are working correctly

        VectorData<int> t_m;
        t_m.resize(m.size());

        uint64_t counter = 0;

        for (size_t i = 0; i < t_m.size(); ++i) {
            counter++;
        }

        ASSERT_EQ(counter,m.size());

    }

    TEST_F(VectorDataTest, resize) {
        // Check initialize and resize and size are working correctly

        //shrinking the array with not-reallocate memory when using resize
        m.resize(sz/2);

        uint64_t counter = 0;

        for (size_t i = 0; i < m.size(); ++i) {
            m[i] = i;
            counter++;
        }

        ASSERT_EQ(counter,m.size());

        //however growing larger then that will re-allocate the memory
        m.resize(sz*2);

        counter = 0;

        for (size_t i = 0; i < m.size(); ++i) {
            counter++;
        }

        ASSERT_EQ(counter,m.size());


    }

    TEST_F(VectorDataTest, BackTest) {

        ASSERT_EQ(m.back(),sz-1);
    }

    TEST_F(VectorDataTest, CopyTest) {
        // Change type and compare if still OK

        VectorData<int> same_type;
        same_type.copy(m);

        for (size_t i = 0; i < same_type.size(); ++i) {
            ASSERT_EQ(same_type[i],m[i]);
        }

        //also works copying to different types
        VectorData<float> diff_type;
        diff_type.copy(m);

        for (size_t i = 0; i < diff_type.size(); ++i) {
            ASSERT_EQ(diff_type[i],m[i]);
        }


    }

    TEST_F(VectorDataTest, FillTest) {
        // Change type and compare if still OK
        VectorData<float> v1;
        v1.init(m.size());
        float fill_val = 23.2;
        v1.fill(fill_val);

        VectorData<float> v2;
        v2.resize(m.size(),fill_val);

        for (size_t i = 0; i < v2.size(); ++i) {
            ASSERT_EQ(v1[i],v2[i]);
        }


    }

    template<typename T, typename S, typename UnaryOperator>
    void map_const(const VectorData<T>& v1, VectorData<S>& out, UnaryOperator op) {
        v1.map(out, op);
    }

    TEST_F(VectorDataTest, MapTest) {

        VectorData<int> v1;
        v1.copy(m);
        v1.map(v1, [](const int& a){ return a*a; });

        for(size_t i = 0; i < v1.size(); ++i) {
            ASSERT_EQ(v1[i],m[i]*m[i]);
        }

        VectorData<float> v2;
        float val = 3.0f;
        v1.map(v2, [val](const int& a){return a/val;});

        for(size_t i = 0; i < v1.size(); ++i) {
            ASSERT_FLOAT_EQ(v2[i], (m[i]*m[i])/val);
        }

        VectorData<float> v3;
        map_const(v2, v3, [](const float& a){ return a-1;});

        for(size_t i = 0; i < v1.size(); ++i) {
            ASSERT_FLOAT_EQ(v3[i], v2[i]-1);
        }

    }

    template<typename T, typename S, typename U, typename BinaryOperator>
    void zip_const(const VectorData<T>& v1, const VectorData<S>& v2, VectorData<U>& out, BinaryOperator op) {
        v1.zip(v2, out, op);
    }

    TEST_F(VectorDataTest, ZipTest) {

        VectorData<int> v1;
        v1.copy(m);
        v1.zip(m, v1, [](const int& a, const int& b){ return a*(b-1); });

        for(size_t i = 0; i < v1.size(); ++i) {
            ASSERT_EQ(v1[i],m[i]*(m[i]-1));
        }

        VectorData<float> v2;
        v2.copy(m);
        v1.zip(v2, v2, [](const int& a, const float& b){return a/(b+1);});

        for(size_t i = 0; i < v1.size(); ++i) {
            ASSERT_FLOAT_EQ(v2[i], (m[i]*(m[i]-1))/((float)m[i]+1));
        }

        VectorData<float> v3;
        zip_const(m, v2, v3, [](const int& a, const float& b){return a+b;});
        for(size_t i = 0; i < v1.size(); ++i) {
            ASSERT_FLOAT_EQ(v3[i], m[i]+v2[i]);
        }
    }



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

    TEST_F(MeshDataTest, resizeTest) {

        PixelData<int> md;
        md.initWithValue(10,10,10,1);

        //this will not re-allocate
        md.initWithResize(9,9,9);

        for(size_t i = 0; i < md.mesh.size(); ++i) {
            ASSERT_EQ(md.mesh[i],1);
        }

        //this will not re-allocate
        md.initWithResize(10,10,10);

        for(size_t i = 0; i < md.mesh.size(); ++i) {
             ASSERT_EQ(md.mesh[i],1);
        }


    }


    TEST(MeshDataSimpleTest, SizeOverflowTest) {
        // Assign large dimensions, without allocating memory
        PixelData<int> tmp;
        tmp.z_num = 1000000;
        tmp.x_num = 1000000;
        tmp.y_num = 1000000;

        // Check that PixelData::size does not overflow
        size_t sz = tmp.size();
        size_t sz_gt = 4000000000000000000;
        ASSERT_EQ(sz, sz_gt);
    }



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
    INSTANTIATE_TEST_SUITE_P(CopyData, MeshDataParameterTest, ::testing::Values<int>(6, 7, 10000));

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
        downsamplePyramid(m, ds, 3, 1);

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

        bool success = true;

        PixelData<int> m;

        std::vector<int> vals = {0,1,2,3,4,5,6,7};

        std::vector<int> expected = {
                7, 6, 7, 6,
                5, 4, 5, 4,
                7, 6, 7, 6,
                5, 4, 5, 4,

                3, 2, 3, 2,
                1, 0, 1, 0,
                3, 2, 3, 2,
                1, 0, 1, 0,

                7, 6, 7, 6,
                5, 4, 5, 4,
                7, 6, 7, 6,
                5, 4, 5, 4,

                3, 2, 3, 2,
                1, 0, 1, 0,
                3, 2, 3, 2,
                1, 0, 1, 0
        };

        PixelData<int> padd_sol;
        padd_sol.init_from_mesh(4,4,4,expected.data());

        m.init_from_mesh(2,2,2,vals.data());

        PixelData<int> padd;

        paddPixels(m, padd, 2, 2, 2);

        //check
        for (size_t i = 0; i < padd_sol.mesh.size(); ++i) {
            int computed_val = padd.mesh[i];
            int expected_val = padd_sol.mesh[i];
            if(computed_val != expected_val){
                success = false;
            }
        }


        PixelData<int> unpadd;

        unpaddPixels(unpadd, padd, 2, 2, 2);

        //check
        for (size_t i = 0; i < unpadd.mesh.size(); ++i) {
            int computed_val = unpadd.mesh[i];
            int expected_val = m.mesh[i];
            if(computed_val != expected_val){
                success = false;
            }
        }

        // quick padd/unpad check with different dims.
        paddPixels(m, padd, 2, 0, 1);
        unpaddPixels(unpadd, padd, 2, 2, 2);

        //check
        for (size_t i = 0; i < unpadd.mesh.size(); ++i) {
            int computed_val = unpadd.mesh[i];
            int expected_val = m.mesh[i];
            if(computed_val != expected_val){
                success = false;
            }
        }

        // quick padd/unpad check with different dims.
        paddPixels(m, padd, 0, 2, 1);
        unpaddPixels(unpadd, padd, 2, 2, 2);

        //check
        for (size_t i = 0; i < unpadd.mesh.size(); ++i) {
            int computed_val = unpadd.mesh[i];
            int expected_val = m.mesh[i];
            if(computed_val != expected_val){
                success = false;
            }
        }

        // quick padd/unpad check with different dims.
        paddPixels(m, padd, 1, 2, 0);
        unpaddPixels(unpadd, padd, 2, 2, 2);

        //check
        for (size_t i = 0; i < unpadd.mesh.size(); ++i) {
            int computed_val = unpadd.mesh[i];
            int expected_val = m.mesh[i];
            if(computed_val != expected_val){
                success = false;
            }
        }

        ASSERT_TRUE(success);
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
