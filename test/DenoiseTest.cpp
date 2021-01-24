//
// Created by bevan on 06/12/2020.
//
#include <gtest/gtest.h>

#include "TestTools.hpp"

#include "data_structures/APR/particles/LazyData.hpp"

#include "numerics/APRDenoise.hpp"

struct TestData{

    APRStencils aprStencils;
    APR apr;
    ParticleData<uint16_t> parts;

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

  testData.aprStencils.stencils.resize(testData.aprStencils.number_levels + 1);

  for(int level = 0;level <= testData.aprStencils.number_levels;level++){

      auto &stencil = testData.aprStencils.stencils[level];

      stencil.stencil_dims.resize(3,1);

      for(int d = 0; d < testData.aprStencils.dim; d++) {
        stencil.stencil_dims[d] = d + 1 + level;
      }

      int num_pts = (2*stencil.stencil_dims[0]+1)*(2*stencil.stencil_dims[1]+1)*(2*stencil.stencil_dims[2]+1);

      stencil.linear_coeffs.resize(num_pts);

      //make the pts just be the index of the array for easy checking;
      for(size_t i = 0; i < stencil.linear_coeffs.size();i++) {
        stencil.linear_coeffs[i] = i;
      }

  }

}


void init_stencil_center(TestData& testData){

  testData.aprStencils.stencils.resize(testData.aprStencils.number_levels + 1);

  for(int level = 0;level <= testData.aprStencils.number_levels;level++){

    auto &stencil = testData.aprStencils.stencils[level];

    stencil.stencil_dims.resize(3,1);

    for(int d = 0; d < testData.aprStencils.dim; d++){
      stencil.stencil_dims[d] = d + 1 + level;
    }

    int num_pts = (2*stencil.stencil_dims[0]+1)*(2*stencil.stencil_dims[1]+1)*(2*stencil.stencil_dims[2]+1);


    int x_num = (2*stencil.stencil_dims[1]+1);
    int y_num = (2*stencil.stencil_dims[0]+1);


    stencil.linear_coeffs.resize(num_pts,0);

    uint64_t center_index = stencil.stencil_dims[0] + stencil.stencil_dims[1]*y_num + stencil.stencil_dims[2]*x_num*y_num;

    stencil.linear_coeffs[center_index] = 1;

  }

}

void Stencil2D::SetUp(){

    testData.aprStencils.dim = 2;
    testData.aprStencils.number_levels = 4;

    std::string file_name = get_source_directory_apr() + "files/Apr/sphere_2D/sphere_2D.apr";

    APRFile aprFile;
    aprFile.open(file_name,"READ");
    aprFile.set_read_write_tree(false);
    aprFile.read_apr(testData.apr);
    aprFile.read_particles(testData.apr,"particle_intensities",testData.parts);
    aprFile.close();

}

void Stencil1D::SetUp(){

  testData.aprStencils.dim = 1;
  testData.aprStencils.number_levels = 6;

  std::string file_name = get_source_directory_apr() + "files/Apr/sphere_1D/sphere_1D.apr";

  APRFile aprFile;
  aprFile.open(file_name,"READ");
  aprFile.set_read_write_tree(false);
  aprFile.read_apr(testData.apr);
  aprFile.read_particles(testData.apr,"particle_intensities",testData.parts);
  aprFile.close();


}

void Stencil3D::SetUp(){

  testData.aprStencils.dim = 3;
  testData.aprStencils.number_levels = 5;



  std::string file_name = get_source_directory_apr() + "files/Apr/sphere_120/sphere.apr";

  APRFile aprFile;
  aprFile.open(file_name,"READ");
  aprFile.set_read_write_tree(false);
  aprFile.read_apr(testData.apr);
  aprFile.read_particles(testData.apr,"particle_intensities",testData.parts);
  aprFile.close();

}



bool test_io(TestData &testData){

  init_stencil(testData);

  bool success = true;

  auto &inputStencils = testData.aprStencils;

  std::string stencil_file_name = "test_all_" + std::to_string(inputStencils.dim) + ".stencils";

  inputStencils.write_stencil(stencil_file_name);

  APRStencils readStencils;

  readStencils.read_stencil(stencil_file_name);

  for(int level = 0; level <= testData.aprStencils.number_levels; level++) {

    auto &stencil_input = inputStencils.stencils[level];
    auto &stencil_read = readStencils.stencils[level];

    for (size_t i = 0; i < stencil_input.linear_coeffs.size(); i++) {

      if (stencil_read.linear_coeffs[i] != stencil_input.linear_coeffs[i]) {
        success = false;
      }

    }
  }

  return success;
}

bool test_apply(TestData &testData){
  //
  //  First test logic, if the particles are all zeros, the correct stencil should just be the sum of all its values.
  //

  init_stencil(testData);

  bool success = true;

  ParticleData<float> onesParticles;
  onesParticles.init(testData.apr);

  ParticleData<float> particlesOutput;
  particlesOutput.init(testData.apr);

  std::fill(onesParticles.begin(),onesParticles.end(),1);

  //load in an APR
  APRDenoise aprDenoise; //should these then be static methods?

  aprDenoise.apply_denoise(testData.apr,onesParticles,particlesOutput,testData.aprStencils);

  auto it = testData.apr.iterator();

  int number_stencils = testData.aprStencils.number_levels;

  for (int level = it.level_max(); level >= it.level_min(); --level) {

    double stencil_total=0;

    auto& stencil = testData.aprStencils.stencils[number_stencils];

    for (size_t i = 0; i < stencil.linear_coeffs.size(); ++i) {
      stencil_total += stencil.linear_coeffs[i];
    }

    uint64_t counter = 0;

    for (auto z = 0; z < it.z_num(level); z++) {
      for (auto x = 0; x < it.x_num(level); ++x) {
        for (it.begin(level, z, x); it < it.end(); it++) {

          double particle_val = particlesOutput[it];

          if(particle_val != stencil_total){
            success = false;
          }
          counter++;

        }
      }
    }

    if(number_stencils > 0){
      number_stencils--;
    }

  }

  return success;
}

bool test_apply_center(TestData &testData){
  //
  //  First test logic, if the particles are all zeros, the correct stencil should just be the sum of all its values.
  //

  init_stencil_center(testData);

  bool success = true;

  ParticleData<float> indexParticles;
  indexParticles.init(testData.apr);

  ParticleData<float> particlesOutput;
  particlesOutput.init(testData.apr);

  for (size_t j = 0; j < indexParticles.size(); ++j) {
    indexParticles[j] = j;
  }

  //load in an APR
  APRDenoise aprDenoise; //should these then be static methods?

  aprDenoise.apply_denoise(testData.apr,indexParticles,particlesOutput,testData.aprStencils);

  auto it = testData.apr.iterator();

  int number_stencils = testData.aprStencils.number_levels;

  for (int level = it.level_max(); level >= it.level_min(); --level) {

    uint64_t counter = 0;

    for (auto z = 0; z < it.z_num(level); z++) {
      for (auto x = 0; x < it.x_num(level); ++x) {
        for (it.begin(level, z, x); it < it.end(); it++) {

          double stencil_total = indexParticles[it];
          double particle_val = particlesOutput[it];

          if(particle_val != stencil_total){
            success = false;
          }
          counter++;

        }
      }
    }

    if(number_stencils > 0){
      number_stencils--;
    }

  }

  return success;
}

bool test_train(TestData& testData){

  bool success = true;

  //load in an APR
  APRDenoise aprDenoise;

  aprDenoise.iteration_max = 200;
  aprDenoise.iteration_others = 100;
  aprDenoise.others_level = 1;

  ParticleData<float> particlesOutput;


  aprDenoise.train_denoise(testData.apr,testData.parts,testData.aprStencils);

  aprDenoise.apply_denoise(testData.apr,testData.parts,particlesOutput,testData.aprStencils);

  float s_threshold = 0.1;

  //output and check stencils
    for (size_t i = 0; i < testData.aprStencils.stencils.size(); ++i) {

        auto stencil = testData.aprStencils.stencils[i];
        float sum = 0;
        for (size_t j = 0; j < stencil.linear_coeffs.size(); ++j) {
            sum+= stencil.linear_coeffs[j];
        }
        if(stencil.linear_coeffs.size() > 0) {
            if (std::abs(sum - 1.0f) > s_threshold) {
                success = false;
            }
        }

    }


  float threshold = 0.2; //arbitrary testing threshold, is the value somewhere close?

  for (size_t i = 0; i < testData.parts.size(); ++i) {

    float diff = std::abs((testData.parts[i] - particlesOutput[i]));

    if(std::isnan(particlesOutput[i])){
        std::cout << "Nan stencil value" << std::endl;
        success = false;
    }

    uint16_t val = particlesOutput[i];

    if(val==0){
        // this should not be produced for these test data.
        std::cout << "Zero result value" << std::endl;
        success = false;
    }

    if((diff / (1.0f*testData.parts[i])) > threshold){
        std::cout << "Result not converged" << std::endl;
        success = false;
    }


  }


  return success;
}


//3D
TEST_F(Stencil3D, Test_IO) {

    ASSERT_TRUE(test_io(testData));

}

TEST_F(Stencil3D, Test_APPLY) {

  ASSERT_TRUE(test_apply(testData));

}

TEST_F(Stencil3D, Test_APPLY_CENTER) {

  ASSERT_TRUE(test_apply_center(testData));

}

TEST_F(Stencil3D, Test_TRAIN) {

  ASSERT_TRUE(test_train(testData));

}

//2D
TEST_F(Stencil2D, Test_IO) {

  ASSERT_TRUE(test_io(testData));

}


TEST_F(Stencil2D, Test_APPLY) {

  ASSERT_TRUE(test_apply(testData));

}


TEST_F(Stencil2D, Test_APPLY_CENTER) {

  ASSERT_TRUE(test_apply_center(testData));

}

TEST_F(Stencil2D, Test_TRAIN) {

  ASSERT_TRUE(test_train(testData));

}


//1D
TEST_F(Stencil1D, Test_IO) {

  ASSERT_TRUE(test_io(testData));

}

TEST_F(Stencil1D, Test_APPLY) {

  ASSERT_TRUE(test_apply(testData));

}

TEST_F(Stencil1D, Test_APPLY_CENTER) {

  ASSERT_TRUE(test_apply_center(testData));

}

TEST_F(Stencil1D, Test_TRAIN) {

  ASSERT_TRUE(test_train(testData));

}




int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}


