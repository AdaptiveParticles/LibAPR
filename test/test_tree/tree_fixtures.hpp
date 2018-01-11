//
// Created by msusik on 04.10.16.
//

#ifndef PARTPLAY_TREE_FIXTURES_HPP
#define PARTPLAY_TREE_FIXTURES_HPP

#include "benchmarks/development/Tree/Tree.hpp"

#include <gtest/gtest.h>

class CreateTreeTest : public ::testing::Test {
public:
    Particle_map<float> particle_map;
    std::vector<uint64_t> tree_mem;
    std::vector<Content> contents_mem;
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};
};

class CreateSmallTreeTest : public CreateTreeTest
{
public:
    void SetUp() override;
};

class CreateBigTreeTest : public CreateTreeTest
{
public:
    void SetUp() override;
};

class CreateNarrowTreeTest : public CreateTreeTest
{
public:
    void SetUp() override;
    
};





#endif //PARTPLAY_TREE_FIXTURES_HPP