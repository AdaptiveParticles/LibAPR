////////////////////////
//
//  Mateusz Susik 2016
//
//  Benchmarking test code
//
////////////////////////

#ifndef PARTPLAY_TESTS_H
#define PARTPLAY_TESTS_H

#include <functional>
#include <gtest/gtest.h>

#include "../src/data_structures/structure_parts.h"
#include "../src/data_structures/meshclass.h"
#include "../src/algorithm/gradient.hpp"
#include "../src/algorithm/variance.hpp"


#define SIZE 100
#define PROFILING 0

typedef std::tuple<std::string, std::string, std::string, float> filepaths_lambda;

class CreateImageTest : public ::testing::Test {
public:
    Part_timer timer;
    Part_rep p_rep;


    Mesh_data<uint16_t> create_test_empty(bool variance);
    Mesh_data<uint16_t> create_bspline_empty();
    Mesh_data<uint16_t> create_variance_empty();
    std::string tests_directory;
protected:

    virtual void SetUp();

    virtual void TearDown() {}
};

class CreateImageFromFileTest : public CreateImageTest, public ::testing::WithParamInterface<filepaths_lambda> {

public:
    Mesh_data<uint16_t> create_test(std::string image_path, bool variance);
    Mesh_data<uint16_t> create_bspline(std::string image_path);
    Mesh_data<uint16_t> create_variance(std::string image_path);
};


#endif //PARTPLAY_TESTS_H
