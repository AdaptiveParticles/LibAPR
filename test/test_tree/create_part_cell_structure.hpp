//
// Bevan Cheeseman 2016
//

#ifndef PARTPLAY_CREATE_PARTCELL_HPP
#define PARTPLAY_CREATE_PARTCELL_HPP

#include "../../src/data_structures/Tree/PartCellStructure.hpp"
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
};



#endif //PARTPLAY_TREE_FIXTURES_HPP