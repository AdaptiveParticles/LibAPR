#include <gtest/gtest.h>

#include "algorithm/LocalParticleCellSet.hpp"
#include "algorithm/PullingScheme.hpp"
#include "algorithm/APRConverter.hpp"
#include "data_structures/APR/access/LinearAccessCuda.hpp"

#include "TestTools.hpp"

namespace {
    template<typename DataType>
    void fillPS(PullingScheme &aPS, PixelData<DataType> &levels) {
        PixelData<DataType> levelsDS(ceil(levels.y_num / 2.0), ceil(levels.x_num / 2.0), ceil(levels.z_num / 2.0));
        LocalParticleCellSet().get_local_particle_cell_set(aPS, levels, levelsDS, APRParameters());
    }

/**
 * Prints PCT
 * @param particleCellTree
 */
    template<typename T>
    void printParticleCellTree(const std::vector<PixelData<T>> &particleCellTree) {
        for (uint64_t l = 0; l < particleCellTree.size(); ++l) {
            auto &tree = particleCellTree[l];
            tree.printMeshT(3, 0);
        }
    }

    /**
     * Create PCT with provided data
     * @param aprInfo
     * @param levels complete list of values from level min to level max in form { {level, min, values}, ..., {level, max, values} }
     *               if levels are not provided PCT with EMPTY values is returned
     * @return Particle Cell Tree with values (or with EMPTY if levels are not provided)
     */
    auto makePCT(const GenInfo &aprInfo, std::initializer_list<std::initializer_list<int>> levels) {
        auto pct = PullingScheme::generateParticleCellTree(aprInfo);

        // Fill particle cell tree only if levels provided - otherwise return tree with EMPTY values
        if (levels.size() != 0) {

            int l = aprInfo.l_min;
            // PS levels range is [l_max - 1, l_min]
            if (((aprInfo.l_max - 1) - aprInfo.l_min + 1) != (int) levels.size()) {
                throw std::runtime_error("Wrong number of level data provided!");
            }
            for (auto &level: levels) {
                if (pct[l].getDimension().size() != level.size()) {
                    std::cerr << "Provided data for level=" << l << " differs from level size "
                              << pct[l].getDimension().size() << " vs. " << level.size() << std::endl;
                    std::cerr << aprInfo << std::endl;
                    throw std::runtime_error("Not this time...");
                }
                std::copy(level.begin(), level.end(), pct[l].mesh.begin());
                l++;
            }
        }
        return pct;
    }

    // Copy PCT - copies only existing levels of it.
    auto copyPCT(const std::vector<PixelData<uint8_t>> &pct) {
        std::vector<PixelData<uint8_t>> copy;
        copy.resize(pct.size());

        for (size_t l = 0; l < pct.size(); ++l) {
            copy[l].initWithResize(pct[l].y_num, pct[l].x_num, pct[l].z_num);
            // Copy only existing levels
            if (pct[l].z_num > 0) copy[l].copyFromMesh(pct[l]);
        }

        return copy;
    }

    // Create random Particle Cell Tree with dimensions specified in 'gi' with given number of particles.
    auto makeRandomPCT(const GenInfo &gi, int numOfParticles = 3) {
        PullingScheme ps;
        ps.initialize_particle_cell_tree(gi);

        // Generate random levels for PS and OVPC
        PixelData<float> levels(std::ceil(gi.org_dims[0]/2.0),
                                std::ceil(gi.org_dims[1]/2.0),
                                std::ceil(gi.org_dims[2]/2.0),
                                0);
        int seed = std::time(nullptr);
        std::srand(seed);
        for (int i = 0; i < numOfParticles; ++i) {
            int modulo = (gi.l_max - gi.l_min);
            if (modulo == 0) modulo = 1;
            levels(std::rand() % levels.y_num, std::rand() % levels.x_num, std::rand() % levels.z_num) = std::rand() % modulo + gi.l_min;
        }
        fillPS(ps, levels);
        ps.pulling_scheme_main();

        return copyPCT(ps.getParticleCellTree());
    }

}

// TODO: There are still problems with computing of small (like 1D images in pipeline)
//       belows test can be used to trigger those errors - should be fixed

//TEST(LinearAccessCudaTest, DeleteMeAfterDevelopment_fullAprPipeline) {
//    // TODO: delete me after development
//    // Full 'get apr' pipeline to test imp. on different stages
//    // Useful during debugging and can be removed once finished
//
//    // Prepare input data (image)
//    int values[] = {9,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
//    // PS input values = 5  0  0  0  0  0  0  0
//
////         int values[] = {3,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 3,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, };
////         PullingScheme input values (local_scale_temp) for above 'image' = {6  0  0  0  0  0  0  0  6  0  0  0  0  0  0  0};
//
//    int len = sizeof(values)/sizeof(int);
//    PixelData<int> data(len, 1, 1);
//    initFromZYXarray(data, values);
//    std::cout << "----- Input image:\n";
//    data.printMeshT(3, 1);
//
//    // Produce APR
//    APR apr;
//    APRConverter<uint16_t> aprConverter;
//    aprConverter.par.rel_error = 0.1;
//    aprConverter.par.lambda = 0.1;
//    aprConverter.par.sigma_th = 0.0001;
//    aprConverter.par.neighborhood_optimization = true;
//    aprConverter.get_apr(apr, data);
//
//    // Print information about APR and all particles
//    std::cout << "APR level min/max: " << apr.level_max() << "/" << apr.level_min() << std::endl;
//    for (int l = apr.level_min(); l <= apr.level_max(); ++l) {
//        std::cout << "    level[" << l << "] size: " << apr.level_size(l) << std::endl;
//    }
//    std::cout << "APR particles z x y level:\n";
//    auto it = apr.iterator();
//    for (int level = it.level_min(); level <= it.level_max(); ++level) {
//        for (int z = 0; z < it.z_num(level); z++) {
//            for (int x = 0; x < it.x_num(level); ++x) {
//                for (it.begin(level, z, x); it < it.end(); it++) {
//                    std::cout << "              " << z << " " << x << " " << it.y() << " " << level << std::endl;
//                }
//            }
//        }
//    }
//    std::cout << std::endl;
//
//    // Sample input
//    ParticleData<uint16_t> particleIntensities;
//    particleIntensities.sample_image(apr, data);
//
//    // Reconstruct image from particles
//    PixelData<uint16_t> reconstructImg;
//    APRReconstruction::reconstruct_constant(apr, reconstructImg, particleIntensities);
//    std::cout << "----- Reconstructed image:"<<std::endl;
//    reconstructImg.printMeshT(3, 1);
//
//    // Show level assigned to each pixel in reconstructed image
//    PixelData<uint16_t> levelImg;
//    APRReconstruction::reconstruct_level(apr, levelImg);
//    std::cout << "----- Image levels:" << std::endl;
//    levelImg.printMeshT(3, 1);
//
//    // Show intensities and levels of each particle
//    std::cout << "----- Particle intensities:\n";
//    for (uint64_t i = 0; i < particleIntensities.size(); i++) std::cout << particleIntensities.data[i] << " ";
//    std::cout << std::endl;
//
//    particleIntensities.fill_with_levels(apr);
//
//    std::cout << "----- Particle levels:\n";
//    for (uint64_t  i = 0; i < particleIntensities.size(); i++) std::cout << particleIntensities.data[i] << " ";
//    std::cout << std::endl;
//
//    // Show some general information about generated APR
//    double computational_ratio = (1.0 * apr.org_dims(0) * apr.org_dims(1) * apr.org_dims(2)) / (1.0 * apr.total_number_particles());
//    std::cout << std::endl;
//    std::cout << "#pixels: " << (apr.org_dims(0) * apr.org_dims(1) * apr.org_dims(2)) << " #particles: " << (apr.total_number_particles()) << std::endl;
//    std::cout << "Computational Ratio (Pixels/Particles): " << std::setprecision(2) << computational_ratio << std::endl;
//}


//TEST(LinearAccessCudaTest, DeleteMeAfterDevelopment_PS) {
//    // TODO: delete me after development
//    // Runs PS to test imp. on different stages
//    // Useful during debugging and can be removed once finished
////    int values[] = {0,0,0,5, 0,0,0,0};
////    int len = sizeof(values)/sizeof(int);
//
//    PixelData<int> levels(8, 1, 1, 0);
//    levels(5,0,0) = 1;
//
////    initFromZYXarray(levels, values);
//    std::cout << "---------------\n";
//    levels.printMeshT(3, 1);
//    std::cout << "---------------\n";
//
//    GenInfo gi;
//    const PixelDataDim dim = levels.getDimension();
//    std::cout << "Levels dim: " << dim << std::endl;
//    gi.init(dim.y * 2, dim.x * 1, dim.z * 1); // time two in y-direction since PS container is downsized.
//    std::cout << gi << std::endl;
//
//    APRTimer t(false);
//
//    t.start_timer("PS1");
//    PullingScheme ps;
//    ps.initialize_particle_cell_tree(gi);
//    int l_max = gi.l_max - 1;
//    int l_min = gi.l_min;
//    std::cout << "PS: max/max min/min" << l_max << " " << ps.pct_level_max() << "  " << l_min << " " << ps.pct_level_min() << std::endl;
//
//    fillPS(ps, levels);
//
//    std::cout << "---------- Filled PS tree\n";
//    printParticleCellTree(ps.getParticleCellTree());
//    std::cout << "---------------\n";
//
//    ps.pulling_scheme_main();
//    t.stop_timer();
//
//    // Useful during debugging and can be removed once finished
//    std::cout << "----------PS:\n";
//    printParticleCellTree(ps.getParticleCellTree());
//    std::cout << "-------------\n";
//
//    LinearAccess linearAccess;
//    linearAccess.genInfo = &gi;
//    APRParameters par;
//    par.neighborhood_optimization = true;
//    linearAccess.initialize_linear_structure(par, ps.getParticleCellTree());
//
//    std::cout << gi << std::endl;
//    auto prt = [&](const auto& v){ std::cout << "size=" << v.size() << " data="; for (size_t i = 0; i < v.size(); i++) std::cout << v[i] << ", "; std::cout << std::endl; };
//    prt(linearAccess.y_vec);
//    prt(linearAccess.xz_end_vec);
//    prt(linearAccess.level_xz_vec);
//
//    LinearIterator it(linearAccess, gi);
//    for (int l = 0; l <= 3; l++) {
//        std::cout << it.particles_level_begin(l) << " " << it.particles_level_end(l) << std::endl;
//    }
//    std::cout << "NumOfParticles: " << gi.total_number_particles << std::endl;
//
//    std::cout << "===========================\n";
//    for (int level = it.level_min(); level <= it.level_max(); ++level) {
//        for (int z = 0; z < it.z_num(level); z++) {
//            for (int x = 0; x < it.x_num(level); ++x) {
//                for (it.begin(level, z, x); it < it.end(); it++) {
//                    std::cout << "              " << z << " " << x << " " << it.y() << " " << level << std::endl;
//                }
//            }
//        }
//    }
//    std::cout << std::endl;
//}

// *********************************************************************************************************************
// Tests of CUDA implementation of LinearAccess
// *********************************************************************************************************************


TEST(LinearAccessCudaTest, optimizationForSmallLevels) {
    // Tests optimized part of LinearAccess returning full-resolution for levels <= 2

    // --- Create input data structures and objects
    GenInfo gi;
    gi.init(4, 3, 2);
    auto pct = makePCT(gi, {}); // In that case values of PCT are not important  (all dense particle data will be generated anyway)

    APRParameters par;
    par.neighborhood_optimization = true;

    // --- Method under test
    auto linearAccess = initializeLinearStructureCuda(gi, par, pct);

    // ---- Verify output
    std::vector<uint16_t> expected_y_vec = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}; // all 'y' particles for each xz
    std::vector<uint64_t> expected_xz_end_vec = {0, 0, 0, 4, 8, 12, 16, 20, 24};
    std::vector<uint64_t> expected_level_xz_vec = {1, 1, 3, 9};

    EXPECT_EQ(compareParticles(expected_y_vec, linearAccess.y_vec), 0);
    EXPECT_EQ(compareParticles(expected_xz_end_vec, linearAccess.xz_end_vec), 0);
    // Useful during debugging and can be removed once finished
    EXPECT_EQ(compareParticles(expected_level_xz_vec, linearAccess.level_xz_vec), 0);

    EXPECT_EQ(gi.total_number_particles, expected_y_vec.size());
    EXPECT_EQ(gi.total_number_particles, 4 * 3 * 2);
}

TEST(LinearAccessCudaTest, optimizationForSmallLevelsVScpu) {
    // Tests optimized part of LinearAccess returning full-resolution for levels <= 2 for all possible combination of xyz
    // For bigger xyz 'optimized' part of code is not used

    for (int x = 1; x <= 4; ++x) {
        for (int y = 1; y <= 4; ++y) {
            for (int z = 1; z <= 4; ++z) {
//                std::cout << "< ============================================= " << x << " " << y << " "<< z << std::endl;
                // --- Create input data structures and objects
                GenInfo gi;
                gi.init(y, x, z);

                auto pct = makePCT(gi, {}); // In that case values of PCT are not important  (all dense particle data will be generated anyway)
                GenInfo giGpu;
                giGpu.init(y, x, z);
                auto pctGpu = makePCT(giGpu, {}); // In that case values of PCT are not important  (all dense particle data will be generated anyway)

                LinearAccess linearAccess;
                linearAccess.genInfo = &gi;
                APRParameters par;
                par.neighborhood_optimization = true;

                // --- Method under test
                linearAccess.initialize_linear_structure(par, pct);
                auto linearAccessGpu = initializeLinearStructureCuda(giGpu, par, pctGpu);

                EXPECT_EQ(compareParticles(linearAccessGpu.y_vec, linearAccess.y_vec), 0);
                EXPECT_EQ(compareParticles(linearAccessGpu.xz_end_vec, linearAccess.xz_end_vec), 0);
                EXPECT_EQ(compareParticles(linearAccessGpu.level_xz_vec, linearAccess.level_xz_vec), 0);

                EXPECT_EQ(giGpu.total_number_particles, gi.total_number_particles);
                EXPECT_EQ(linearAccessGpu.y_vec.size(), linearAccess.y_vec.size());
            }
        }
    }

}

TEST(LinearAccessCudaTest, testGPUvsCPUforDifferentSizes) {

    for (int x : {1, 2, 4, 100, 255}) {
        for (int y : {1, 2, 4, 100, 256}) {
            for (int z : {1, 2, 4, 100, 257}) {
//                std::cout << "< ============================================= " << y << " " << x << " "<< z << std::endl;

                // ----------- Create input data structures and objects
                GenInfo gi;
                gi.init(y, x, z);

                auto pct = makeRandomPCT(gi, 133);

                auto pctCpu = copyPCT(pct);
                auto pctGpu = copyPCT(pct);

                GenInfo giGpu;
                giGpu.init(y, x, z);

                LinearAccess linearAccess;
                linearAccess.genInfo = &gi;
                APRParameters par;
                par.neighborhood_optimization = true;


                // --------- methods under test
                APRTimer t(false);
                t.start_timer("__________________________ CPU");
                // --- Method under test
                linearAccess.initialize_linear_structure(par, pctCpu);
                t.stop_timer();

                t.start_timer("_________________________ GPU");
                auto linearAccessGpu = initializeLinearStructureCuda(giGpu, par, pctGpu);
                t.stop_timer();


                // ----------- verify results

                // LinearAccess changes PCT - compare if changes in CPU and GPU side are same
                EXPECT_EQ(compareParticleCellTrees(pctCpu, pctGpu), 0);

                // Test if returned structures have same data
                EXPECT_EQ(compareParticles(linearAccessGpu.y_vec, linearAccess.y_vec), 0);
                EXPECT_EQ(compareParticles(linearAccessGpu.level_xz_vec, linearAccess.level_xz_vec), 0);
                EXPECT_EQ(compareParticles(linearAccessGpu.y_vec, linearAccess.y_vec), 0);

                EXPECT_EQ(giGpu.total_number_particles, gi.total_number_particles);
                EXPECT_EQ(linearAccessGpu.y_vec.size(), linearAccess.y_vec.size());
            }
        }
    }

}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
