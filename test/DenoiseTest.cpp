//
// Created by bevan on 06/12/2020.
//
#include <gtest/gtest.h>

#include "TestTools.hpp"

#include "data_structures/APR/particles/LazyData.hpp"

#include "numerics/APRDenoise.hpp"

struct TestData{

    APRStencils aprStencils;
    int dim;
    int l_max;
    int l_min;

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


void init_stencil(TestData& testData){

  testData.aprStencils.dim = testData.dim;

  testData.aprStencils.level_min = testData.l_max;
  testData.aprStencils.level_max = testData.l_min;

  testData.aprStencils.stencils.resize(testData.l_max + 1);

  for(int level = testData.l_min;level <= testData.l_max;level++){

      auto &stencil = testData.aprStencils.stencils[level];

      stencil.stencil_dims.resize(3,1);

      for(int d = 0; d < testData.dim; d++){
          stencil.stencil_dims[d] = d + 1 + level;
          int num_pts = (2*stencil.stencil_dims[0]+1)*(2*stencil.stencil_dims[1]+1)*(2*stencil.stencil_dims[2]+1);

          stencil.linear_coeffs.resize(num_pts);

          //make the pts just be the index of the array for easy checking;
          for(int i = 0; i < stencil.linear_coeffs.size();i++){
            stencil.linear_coeffs[i] = i;
          }
      }


  }



}

void Stencil3D::SetUp(){

    testData.dim = 3;
    testData.l_max = 4;
    testData.l_min = 2;

    init_stencil(testData);


}




bool test_io(TestData &testData){

  bool success = true;

  auto &as = testData.aprStencils;

  std::string stencil_file_name = "test";
  Stencil<double> stencil_read;

  as.write_stencil(stencil_file_name,as.stencils.back());

  as.read_stencil(stencil_file_name,stencil_read);

  auto &stencil_input = as.stencils.back();

  for(int i = 0; i < stencil_input.linear_coeffs.size();i++){

    if(stencil_read.linear_coeffs[i] != stencil_input.linear_coeffs[i]){
      success = false;
    }

  }

  return success;
}




//1D
TEST_F(Stencil3D, EXAMPLE_DENOISE) {

    ASSERT_TRUE(test_io(testData));

}





int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}


