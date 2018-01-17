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

#include "benchmarks/development/old_structures/structure_parts.h"
#include "src/data_structures/Mesh/MeshData.hpp"
#include "benchmarks/development/old_algorithm/gradient.hpp"
#include "benchmarks/development/old_algorithm/variance.hpp"


#define SIZE 100
#define PROFILING 0

typedef std::tuple<std::string, std::string, std::string, float> filepaths_lambda;

class CreateImageTest : public ::testing::Test {
public:
    Part_timer timer;
    Part_rep p_rep;


    MeshData<uint16_t> create_test_empty(bool variance);
    MeshData<uint16_t> create_bspline_empty();
    MeshData<uint16_t> create_variance_empty();
    std::string tests_directory;
protected:

    virtual void SetUp();

    virtual void TearDown() {}
};

class CreateImageFromFileTest : public CreateImageTest, public ::testing::WithParamInterface<filepaths_lambda> {

public:
    MeshData<uint16_t> create_test(std::string image_path, bool variance);
    MeshData<uint16_t> create_bspline(std::string image_path);
    MeshData<uint16_t> create_variance(std::string image_path);
};


#endif //PARTPLAY_TESTS_H
