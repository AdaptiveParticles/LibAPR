//
// Bevan Cheeseman 2016
//

#ifndef PARTPLAY_CREATE_PARTCELL_HPP_Q
#define PARTPLAY_CREATE_PARTCELL_HPP_Q

#include "benchmarks/development/Tree/PartCellStructure.hpp"
#include "../../src/io/partcell_io.h"

#include "../utils.h"

#include <gtest/gtest.h>

class CreatePartCellTest : public ::testing::Test {
public:
    PartCellStructure<float,uint64_t> pc_struct;
    Particle_map<float> particle_map;
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};


};

class CreateSphereTest : public CreatePartCellTest
{
public:
    void SetUp() override;
    void TestBody() override;
};

class CreateMembraneTest : public CreatePartCellTest
{
public:
    void SetUp() override;
};

void CreateSphereTest::TestBody(){}

void CreateSphereTest::SetUp(){

    std::string name = "files/partcell_files/test_sphere1_pcstruct_part.h5";

    create_test_dataset_from_hdf5(particle_map,pc_struct,name);

}

void CreateMembraneTest::SetUp(){

    std::string name = "files/partcell_files/membrane_pcstruct_part.h5";

    create_test_dataset_from_hdf5(particle_map,pc_struct,name);

}


#endif //PARTPLAY_TREE_FIXTURES_HPP