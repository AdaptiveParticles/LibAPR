//
// Created by msusik on 25.08.16.
//

#ifndef PARTPLAY_K_TEST_H
#define PARTPLAY_K_TEST_H

#include <gtest/gtest.h>
#include "../src/algorithm/level.hpp"

#include "../src/data_structures/structure_parts.h"
#include "../src/data_structures/meshclass.h"

typedef std::tuple<std::string, std::string> grad_and_var_paths;

class CreateKTest : public ::testing::Test {
public:
    Part_timer timer;
    Part_rep p_rep;

    std::string tests_directory;
protected:

    virtual void SetUp();

    virtual void TearDown() {}
};

class CreateKFromFilesTest : public CreateKTest, public ::testing::WithParamInterface<grad_and_var_paths > {

public:
    Particle_map<float> create_k(std::string grad_path, std::string output_path);
};


#endif //PARTPLAY_K_TEST_H
