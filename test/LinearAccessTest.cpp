#include <gtest/gtest.h>

#include "algorithm/PullingScheme.hpp"
#include "algorithm/LocalParticleCellSet.hpp"

#include "TestTools.hpp"


/**
 * Create PCT with provided data
 * @param aprInfo
 * @param levels complete list of values from level min to level max in form { {level, min, values}, ..., {level, max, values} }
 * @return Particle Cell Tree with values
 */
auto makePCT(const GenInfo &aprInfo, std::initializer_list<std::initializer_list<int>> levels) {
    auto pct = PullingScheme::generateParticleCellTree(aprInfo);


    int l = aprInfo.l_min;

    // PS levels range is [l_max - 1, l_min]
    if (((aprInfo.l_max - 1) - aprInfo.l_min + 1) != (int) levels.size()) {
        throw std::runtime_error("Wrong number of level data provided!");
    }
    for (auto &level : levels) {
        if (pct[l].getDimension().size() != level.size()) {
            std::cerr << "Provided data for level=" << l << " differs from level size " << pct[l].getDimension().size() << " vs. " << level.size() << std::endl;
            std::cerr << aprInfo << std::endl;
            throw std::runtime_error("Not this time...");
        }
        std::copy(level.begin(), level.end(), pct[l].mesh.begin());
        l++;
    }

    return pct;
}

TEST(LinearAccessTest, optimizationForSmallLevels) {

    // --- Create input data structures and objects
    GenInfo gi;
    gi.init(4, 3, 2);
    auto pct = makePCT(gi, {{1, 2, 3, 4}}); // In that case values of PCT are not important  (all dense particle data will be generated anyway)

    LinearAccess linearAccess;
    linearAccess.genInfo = &gi;
    APRParameters par;
    par.neighborhood_optimization = true;

    // --- Method under test
    linearAccess.initialize_linear_structure(par, pct);

    // ---- Verify output
    std::vector<uint16_t> expected_y_vec = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}; // all 'y' particles for each xz
    std::vector<uint64_t> expected_xz_end_vec = {0, 0, 0, 4, 8, 12, 16, 20, 24};
    std::vector<uint64_t> expected_level_xz_vec = {1, 1, 3, 9};

    EXPECT_EQ(compareParticles(expected_y_vec, linearAccess.y_vec), 0);
    EXPECT_EQ(compareParticles(expected_xz_end_vec, linearAccess.xz_end_vec), 0);
    EXPECT_EQ(compareParticles(expected_level_xz_vec, linearAccess.level_xz_vec), 0);
}

TEST(LinearAccessTest, yDirNeighbourhoodOptTrue) {

    // --- Create input data structures and objects
    GenInfo gi;
    gi.init(16, 1, 1);

    auto pct = makePCT(gi, {{0, 0},
                            {0, 0, 3, 3},
                            {1, 2, 3, 3, 0, 0, 0, 0}});

    LinearAccess linearAccess;
    linearAccess.genInfo = &gi;
    APRParameters par;
    par.neighborhood_optimization = true;

    // --- Method under test
    linearAccess.initialize_linear_structure(par, pct);

    // ---- Verify output
    std::vector<uint16_t> expected_y_vec = {2, 3, 1, 2, 3, 0, 1};
    std::vector<uint64_t> expected_xz_end_vec = {0, 0, 2, 5, 7};
    std::vector<uint64_t> expected_level_xz_vec = {1, 1, 2, 3, 4, 5};

    EXPECT_EQ(compareParticles(expected_y_vec, linearAccess.y_vec), 0);
    EXPECT_EQ(compareParticles(expected_xz_end_vec, linearAccess.xz_end_vec), 0);
    EXPECT_EQ(compareParticles(expected_level_xz_vec, linearAccess.level_xz_vec), 0);
}

TEST(LinearAccessTest, yDirNeighbourhoodOptFalse) {

    // --- Create input data structures and objects
    GenInfo gi;
    gi.init(16, 1, 1);

    auto pct = makePCT(gi, {{0, 0},
                            {0, 0, 3, 3},
                            {1, 2, 3, 3, 0, 0, 0, 0}});

    LinearAccess linearAccess;
    linearAccess.genInfo = &gi;
    APRParameters par;
    par.neighborhood_optimization = false;

    // --- Method under test
    linearAccess.initialize_linear_structure(par, pct);

    // ---- Verify output
    std::vector<uint16_t> expected_y_vec = {2, 3, 2, 3, 0, 1, 2, 3};
    std::vector<uint64_t> expected_xz_end_vec = {0, 0, 2, 4, 8};
    std::vector<uint64_t> expected_level_xz_vec = {1, 1, 2, 3, 4, 5};

    EXPECT_EQ(compareParticles(expected_y_vec, linearAccess.y_vec), 0);
    EXPECT_EQ(compareParticles(expected_xz_end_vec, linearAccess.xz_end_vec), 0);
    EXPECT_EQ(compareParticles(expected_level_xz_vec, linearAccess.level_xz_vec), 0);
}

TEST(LinearAccessTest, xDirNeighbourhoodOptTrue) {

    // --- Create input data structures and objects
    GenInfo gi;
    gi.init(1, 16, 1);

    auto pct = makePCT(gi, {{0, 0},
                            {0, 0, 3, 3},
                            {1, 2, 3, 3, 0, 0, 0, 0}});

    LinearAccess linearAccess;
    linearAccess.genInfo = &gi;
    APRParameters par;
    par.neighborhood_optimization = true;

    // --- Method under test
    linearAccess.initialize_linear_structure(par, pct);

    // ---- Verify output
    std::vector<uint16_t> expected_y_vec = {0, 0, 0, 0, 0, 0, 0};
    std::vector<uint64_t> expected_xz_end_vec = {0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    std::vector<uint64_t> expected_level_xz_vec = {1, 1, 3, 7, 15, 31};

    EXPECT_EQ(compareParticles(expected_y_vec, linearAccess.y_vec), 0);
    EXPECT_EQ(compareParticles(expected_xz_end_vec, linearAccess.xz_end_vec), 0);
    EXPECT_EQ(compareParticles(expected_level_xz_vec, linearAccess.level_xz_vec), 0);
}

TEST(LinearAccessTest, xDirNeighbourhoodOptFalse) {

    // --- Create input data structures and objects
    GenInfo gi;
    gi.init(1, 16, 1);

    auto pct = makePCT(gi, {{0, 0},
                            {0, 0, 3, 3},
                            {1, 2, 3, 3, 0, 0, 0, 0}});

    LinearAccess linearAccess;
    linearAccess.genInfo = &gi;
    APRParameters par;
    par.neighborhood_optimization = false;

    // --- Method under test
    linearAccess.initialize_linear_structure(par, pct);

    // ---- Verify output
    std::vector<uint16_t> expected_y_vec = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<uint64_t> expected_xz_end_vec = {0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
    std::vector<uint64_t> expected_level_xz_vec = {1, 1, 3, 7, 15, 31};

    EXPECT_EQ(compareParticles(expected_y_vec, linearAccess.y_vec), 0);
    EXPECT_EQ(compareParticles(expected_xz_end_vec, linearAccess.xz_end_vec), 0);
    EXPECT_EQ(compareParticles(expected_level_xz_vec, linearAccess.level_xz_vec), 0);
}

TEST(LinearAccessTest, zDirNeighbourhoodOptTrue) {

    // --- Create input data structures and objects
    GenInfo gi;
    gi.init(1, 1, 16);

    auto pct = makePCT(gi, {{0, 0},
                            {0, 0, 3, 3},
                            {1, 2, 3, 3, 0, 0, 0, 0}});

    LinearAccess linearAccess;
    linearAccess.genInfo = &gi;
    APRParameters par;
    par.neighborhood_optimization = true;

    // --- Method under test
    linearAccess.initialize_linear_structure(par, pct);

    // ---- Verify output
    std::vector<uint16_t> expected_y_vec = {0, 0, 0, 0, 0, 0, 0};
    std::vector<uint64_t> expected_xz_end_vec = {0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    std::vector<uint64_t> expected_level_xz_vec = {1, 1, 3, 7, 15, 31};

    EXPECT_EQ(compareParticles(expected_y_vec, linearAccess.y_vec), 0);
    EXPECT_EQ(compareParticles(expected_xz_end_vec, linearAccess.xz_end_vec), 0);
    EXPECT_EQ(compareParticles(expected_level_xz_vec, linearAccess.level_xz_vec), 0);
}

TEST(LinearAccessTest, zDirNeighbourhoodOptFalse) {

    // --- Create input data structures and objects
    GenInfo gi;
    gi.init(1, 1, 16);

    auto pct = makePCT(gi, {{0, 0},
                            {0, 0, 3, 3},
                            {1, 2, 3, 3, 0, 0, 0, 0}});

    LinearAccess linearAccess;
    linearAccess.genInfo = &gi;
    APRParameters par;
    par.neighborhood_optimization = false;

    // --- Method under test
    linearAccess.initialize_linear_structure(par, pct);

    // ---- Verify output
    std::vector<uint16_t> expected_y_vec = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<uint64_t> expected_xz_end_vec = {0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
    std::vector<uint64_t> expected_level_xz_vec = {1, 1, 3, 7, 15, 31};

    EXPECT_EQ(compareParticles(expected_y_vec, linearAccess.y_vec), 0);
    EXPECT_EQ(compareParticles(expected_xz_end_vec, linearAccess.xz_end_vec), 0);
    EXPECT_EQ(compareParticles(expected_level_xz_vec, linearAccess.level_xz_vec), 0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}