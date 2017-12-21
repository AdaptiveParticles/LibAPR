//
// Created by msusik on 29.08.16.
//

#ifndef PARTPLAY_RAYTRACE_TEST_H
#define PARTPLAY_RAYTRACE_TEST_H

#include <gtest/gtest.h>
#include "../../src/algorithm/level.hpp"
#include "../../src/data_structures/structure_parts.h"


typedef std::tuple<std::string, std::string, std::string> grad_and_var_paths;

class CreateResultTest : public ::testing::Test {
public:
    Part_timer timer;
    Part_rep p_rep;

    std::string tests_directory;
protected:

    virtual void SetUp();

    virtual void TearDown() {}
};

class RaytraceTest: public CreateResultTest, public ::testing::WithParamInterface<grad_and_var_paths > {

public:
    Particle_map<float> create_result(std::string tiff_path, std::string stats_path, std::string image_name);
};


#endif //PARTPLAY_RAYTRACE_H
