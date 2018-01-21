//
// Created by cheesema on 21.01.18.
//

#include <gtest/gtest.h>
#include "src/data_structures/APR/APR.hpp"
#include "test/utils.h"

class CreateAPRTest : public ::testing::Test {
public:

    APR<float> apr;
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

};

class CreateSmallSphereTest : public CreateAPRTest
{
public:
    void SetUp() override;
};

void CreateSmallSphereTest::SetUp(){

    std::string file_name = "test/files/sphere_120_apr.h5";

    apr.read_apr(file_name);

}

TEST_F(CreateSmallSphereTest, APR_SERIAL_ITERATOR) {

//
//  Sparse Particle Structure Test Cases
//
//

//test iteration
ASSERT_TRUE(utest_apr_serial_iterate(pc_struct));

}

//TEST_F(CreateMembraneTest, APR_SERIAL_NEIGH) {
//
////
////  Sparse Particle Structure Test Cases
////
////
//
////test get face neighbours
//ASSERT_TRUE(utest_apr_serial_neigh(pc_struct));
//
//}
//
//
//TEST_F(CreateMembraneTest, APR_PARALELL_ITERATOR) {
//
////
////  Sparse Particle Structure Test Cases
////
////
//
////test io
//ASSERT_TRUE(utest_apr_parallel_iterate(pc_struct));
//
//}
//
//TEST_F(CreateMembraneTest, APR_PARALELL_NEIGH) {
//
////
////  Sparse Particle Structure Test Cases
////
////
//
////test io
//ASSERT_TRUE(utest_apr_parallel_neigh(pc_struct));
//
//}
//
//TEST_F(CreateMembraneTest, APR_INPUT_OUTPUT) {
//
////
////  Sparse Particle Structure Test Cases
////
////
//
////test io
//ASSERT_TRUE(utest_apr_read_write(pc_struct));
//
//}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}