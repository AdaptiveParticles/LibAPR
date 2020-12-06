//
// Created by bevan on 06/12/2020.
//
//
// Created by bevan on 29/11/2020.
//
#include <gtest/gtest.h>

#include "TestTools.hpp"

#include "data_structures/APR/particles/LazyData.hpp"

#include "io/APRFile.hpp"

#include "numerics/APRDenoise.hpp"

struct TestData{

    std::vector<int> stencil_dims;
    std::vector<double> stencil_data;
    int dim;

};

class CreateDenoiseTest : public ::testing::Test {
public:
    TestData testData;

protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

};

class Stencil3D : public CreateDenoiseTest
{
public:
    void SetUp() override;
};
class Stencil2D : public CreateDenoiseTest
{
public:
    void SetUp() override;
};

class Stencil1D : public CreateDenoiseTest
{
public:
    void SetUp() override;
};


void Stencil3D::SetUp(){

    testData.stencil_dims.resize(3,0);

    testData.stencil_dims[0] = 3;
    testData.stencil_dims[1] = 2;
    testData.stencil_dims[2] = 4;

    testData.dim = 3;

    int num_pts = (2*testData.stencil_dims[0]+1)*(2*testData.stencil_dims[1]+1)*(2*testData.stencil_dims[2]+1);

    testData.stencil_data.resize(num_pts);

    //make the pts just be the index of the array for easy checking;
    for(int i = 0; i < testData.stencil_data.size();i++){
        testData.stencil_data[i] = i;
    }

}



//1D
TEST_F(Stencil3D, EXAMPLE_DENOISE) {

    ASSERT_TRUE(run_denoise_example(test_data));

}





int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}


